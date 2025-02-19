use std::time::Duration;

use clap::Parser;
use futures::StreamExt;
use ndarray::stack;
use ndarray::Array;
use ndarray::Axis;
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::session::SessionOutputs;
use r2r::QosProfile;
use tokio::{join, task};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    dylib: String,

    #[arg(long)]
    model: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_onnx", "")?;

    let pub_cmd_vel = node.create_publisher::<r2r::geometry_msgs::msg::Twist>(
        "/cmd_vel",
        QosProfile::sensor_data(),
    )?;
    let pub_kicker = node
        .create_publisher::<r2r::std_msgs::msg::Bool>("/nv1/kicker", QosProfile::sensor_data())?;

    let mut sub_nv1_speed = node
        .subscribe::<r2r::geometry_msgs::msg::Vector3>("/nv1/speed", QosProfile::sensor_data())?;
    let mut sub_nv1_ir =
        node.subscribe::<r2r::geometry_msgs::msg::Vector3>("/nv1/ir", QosProfile::sensor_data())?;
    let mut sub_nv1_have_ball =
        node.subscribe::<r2r::std_msgs::msg::Bool>("/nv1/have_ball", QosProfile::sensor_data())?;
    let mut sub_nv1_goal_own = node
        .subscribe::<r2r::std_msgs::msg::Float64>("/nv1/opencv/opp", QosProfile::sensor_data())?;

    let mut timer = node
        .create_wall_timer(Duration::from_millis(1000 / 50))
        .unwrap();

    task::spawn(async move {
        ort::init_from(args.dylib)
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .commit()
            .unwrap();

        let model = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(args.model)
            .unwrap();

        const DECISION_PEIROAD: usize = 1;
        let mut decision_counter = 0;

        let mut episode_starts: f32 = 1.0;
        let mut gather1 = Array::from_elem((2, 1, 1, 256), 0.);
        let mut gather2 = Array::from_elem((2, 1, 1, 256), 0.);

        let mut pub_msg_cmd_vel = r2r::geometry_msgs::msg::Twist::default();
        let mut pub_msg_kicker = r2r::std_msgs::msg::Bool::default();

        loop {
            let (_wait_time, sensor_speed, sensor_ir, sensor_have_ball, sensor_goal_own) = join!(
                timer.tick(),
                sub_nv1_speed.next(),
                sub_nv1_ir.next(),
                sub_nv1_have_ball.next(),
                sub_nv1_goal_own.next()
            );

            if sensor_speed.is_none()
                || sensor_ir.is_none()
                || sensor_have_ball.is_none()
                || sensor_goal_own.is_none()
            {
                continue;
            }

            let _sensor_speed = sensor_speed.unwrap();
            let sensor_ir = sensor_ir.unwrap();
            let _sensor_have_ball = sensor_have_ball.unwrap();
            let sensor_goal_own = sensor_goal_own.unwrap();

            let sensor_goal_own = sensor_goal_own.data as f32;
            let sensor_goal_own_x = sensor_goal_own.cos();
            let sensor_goal_own_y = sensor_goal_own.sin();

            // ir x
            // ir y
            let obs = [
                sensor_ir.y as f32,
                sensor_ir.x as f32,
                sensor_goal_own_x,
                sensor_goal_own_y,
            ];

            let mut input = Array::zeros((1, 4));

            for (i, v) in obs.iter().enumerate() {
                input[[0, i]] = *v;
            }

            let outputs: SessionOutputs = model
                .run(
                    ort::inputs!["obs.1" => input.view(), "episode_starts" => Array::from_vec(vec![episode_starts]), "onnx::Gather_1" => gather1.view(), "onnx::Gather_2" => gather2.view()]
                        .unwrap(),
                )
                .unwrap();

            let output = outputs["mean_actions.3"]
                .try_extract_tensor::<f32>()
                .unwrap()
                .t()
                .into_owned();

            let gather1_1 = outputs["137"].try_extract_tensor::<f32>().unwrap();
            let gather1_2 = outputs["138"].try_extract_tensor::<f32>().unwrap();
            let gather2_1 = outputs["246"].try_extract_tensor::<f32>().unwrap();
            let gather2_2 = outputs["247"].try_extract_tensor::<f32>().unwrap();

            gather1.assign(&stack![Axis(0), gather1_1, gather1_2]);
            gather2.assign(&stack![Axis(0), gather2_1, gather2_2]);

            // let rotation = if output[[4, 0]] > 0.5 {
            //     (output[[2, 0]] as f64).atan2(output[[3, 0]])
            // } else {
            //     pub_msg_cmd_vel.angular.z
            // };

            let y = output[[0, 0]] as f64;
            let x = output[[1, 0]] as f64;

            let len = (x * x + y * y).sqrt();
            let norm_x = x / len;
            let norm_y = y / len;

            if decision_counter == DECISION_PEIROAD {
                pub_msg_cmd_vel = r2r::geometry_msgs::msg::Twist {
                    linear: r2r::geometry_msgs::msg::Vector3 {
                        x: norm_x,
                        y: norm_y,
                        z: 0.0,
                    },
                    angular: r2r::geometry_msgs::msg::Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                };

                // pub_msg_kicker = r2r::std_msgs::msg::Bool {
                //     data: output[[4, 0]] > 0.5,
                // };

                pub_msg_kicker = r2r::std_msgs::msg::Bool { data: false };

                decision_counter = 0;
            } else {
                decision_counter += 1;
            }

            pub_cmd_vel.publish(&pub_msg_cmd_vel).unwrap();
            pub_kicker.publish(&pub_msg_kicker).unwrap();

            episode_starts = 0.0;

            println!("cmd_vel: {:#?}", pub_msg_cmd_vel);
            println!("kicker: {:#?}", pub_msg_kicker);
        }
    });

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
