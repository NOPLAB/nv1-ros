use std::f64::consts::PI;

use futures::StreamExt;
use ndarray::Array;
use ort::{GraphOptimizationLevel, Session, SessionOutputs};
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

    task::spawn(async move {
        let dylib_path = "/opt/onnxruntime/build/Linux/Release/libonnxruntime.so";
        ort::init_from(dylib_path).commit().unwrap();

        let model = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file("/home/jetson/robocup/nv1-ros/model_v81_4849994.onnx")
            .unwrap();

        loop {
            let (sensor_speed, sensor_ir) = join!(sub_nv1_speed.next(), sub_nv1_ir.next());
            if sensor_speed.is_none() || sensor_ir.is_none() {
                break;
            }

            let sensor_speed = sensor_speed.unwrap();
            let sensor_ir = sensor_ir.unwrap();

            // velocity x
            // velocity y
            // angle normal cos
            // angle normal sin
            // ir x
            // ir y
            // ir distance
            // have ball
            let msg_array = [
                sensor_speed.x,
                sensor_speed.y,
                (sensor_speed.z + PI / 2.0).cos(),
                (sensor_speed.z + PI / 2.0).sin(),
                sensor_ir.x,
                sensor_ir.y,
                sensor_ir.z,
                1.0,
            ];

            println!("{:?}", msg_array);

            let msg_vec = Vec::from(msg_array);

            let mut input = Array::zeros((1, 8));

            for (i, v) in msg_vec.iter().enumerate() {
                input[[0, i]] = *v as f32;
            }

            let outputs: SessionOutputs = model
                .run(ort::inputs!["obs_0" => input.view()].unwrap())
                .unwrap();

            let output = outputs["continuous_actions"]
                .try_extract_tensor::<f32>()
                .unwrap()
                .t()
                .into_owned();

            let rotation = if output[[2, 0]] > 0.1 {
                let clamped = if output[[2, 0]] > 1.0 {
                    1.0
                } else {
                    output[[2, 0]]
                };
                -clamped as f64
            } else if output[[3, 0]] > 0.1 {
                let clamped = if output[[3, 0]] > 1.0 {
                    1.0
                } else {
                    output[[3, 0]]
                };
                clamped as f64
            } else {
                0.0
            };

            let pub_msg_cmd_vel = &r2r::geometry_msgs::msg::Twist {
                linear: r2r::geometry_msgs::msg::Vector3 {
                    x: output[[0, 0]] as f64,
                    y: output[[1, 0]] as f64,
                    z: 0.0,
                },
                angular: r2r::geometry_msgs::msg::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: rotation * 6.28,
                },
            };

            let pub_msg_kicker = &r2r::std_msgs::msg::Bool {
                data: output[[4, 0]] > 0.5,
            };

            pub_cmd_vel.publish(&pub_msg_cmd_vel).unwrap();
            pub_kicker.publish(&pub_msg_kicker).unwrap();

            println!("cmd_vel: {:#?}", pub_msg_cmd_vel);

            // println!(
            //     "speed: {:?}, ir: {:?}, output: {:?}, kick: {:?}",
            //     sensor_speed, sensor_ir, pub_msg_cmd_vel, pub_msg_kicker
            // );
        }
    });

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
