use std::{collections::VecDeque, time::Duration};

use futures::StreamExt;
use ndarray::Array;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session, SessionOutputs};
use r2r::QosProfile;
use tokio::{join, task};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    let mut timer = node
        .create_wall_timer(Duration::from_millis(1000 / 50))
        .unwrap();
    task::spawn(async move {
        let dylib_path = "/opt/onnxruntime/build/Linux/Release/libonnxruntime.so";
        ort::init_from(dylib_path)
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .commit()
            .unwrap();

        let model = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file("/home/jetson/robocup/nv1-ros/model_v94.onnx")
            .unwrap();

        const COLLECT_VECTOR: usize = 1;

        const DECISION_PEIROAD: usize = 1;
        let mut decision_counter = 0;

        let mut obs_queue = VecDeque::new();
        for _ in 0..COLLECT_VECTOR {
            obs_queue.push_front([0.0, 0.0]);
        }

        let mut pub_msg_cmd_vel = r2r::geometry_msgs::msg::Twist::default();
        let mut pub_msg_kicker = r2r::std_msgs::msg::Bool::default();

        loop {
            let (wait_time, sensor_speed, sensor_ir, have_ball) = join!(
                timer.tick(),
                sub_nv1_speed.next(),
                sub_nv1_ir.next(),
                sub_nv1_have_ball.next()
            );
            if sensor_speed.is_none() || sensor_ir.is_none() || have_ball.is_none() {
                break;
            }

            println!("timer: {:#?}", wait_time.unwrap());

            let _sensor_speed = sensor_speed.unwrap();
            let sensor_ir = sensor_ir.unwrap();
            let _have_ball = have_ball.unwrap();

            // ir x
            // ir y
            let obs = [sensor_ir.x, sensor_ir.y];

            obs_queue.push_back(obs);
            obs_queue.pop_front();

            println!("{:?}", obs);

            let mut input = Array::zeros((1, obs.len() * COLLECT_VECTOR));

            for (i, o) in obs_queue.iter().enumerate() {
                for (j, v) in o.iter().enumerate() {
                    input[[0, i * obs.len() + j]] = *v as f32;
                }
            }

            let outputs: SessionOutputs = model
                .run(ort::inputs!["obs_0" => input.view()].unwrap())
                .unwrap();

            let output = outputs["continuous_actions"]
                .try_extract_tensor::<f32>()
                .unwrap()
                .t()
                .into_owned();

            // let rotation = if output[[4, 0]] > 0.5 {
            //     (output[[2, 0]] as f64).atan2(output[[3, 0]])
            // } else {
            //     pub_msg_cmd_vel.angular.z
            // };

            let x = output[[0, 0]] as f64;
            let y = output[[1, 0]] as f64;

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
