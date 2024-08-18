use futures::stream::StreamExt;
use r2r::QosProfile;

use std::time::Duration;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_onnx", "")?;

    let pub_cmd_vel =
        node.create_publisher::<r2r::geometry_msgs::msg::Twist>("/cmd_vel", QosProfile::default())?;

    task::spawn(async move { loop {} });

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
