use futures::stream::StreamExt;
use r2r::QosProfile;

use std::time::Duration;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_communicator", "")?;

    let mut sub_teleop =
        node.subscribe::<r2r::geometry_msgs::msg::Twist>("/cmd_vel", QosProfile::default())?;

    task::spawn(async move {
        let ports = serialport::available_ports().expect("No ports found!");
        for p in ports {
            println!("{}", p.port_name);
        }

        let mut port = serialport::new("/dev/ttyTHS1", 115200)
            .timeout(Duration::from_millis(10))
            .open()
            .expect("Failed to open port");

        loop {
            match sub_teleop.next().await {
                Some(msg) => {
                    println!("{:#?}", msg);

                    let msgpack = nv1_msg::HubMsgPackRx {
                        vel: nv1_msg::Velocity {
                            linear_x: msg.linear.z as f32,
                            linear_y: msg.linear.x as f32,
                            angular_z: msg.angular.z as f32,
                        },
                        kick: false,
                    };

                    let msgpack_decoded = corepack::to_bytes(msgpack).unwrap();

                    let mut cobs_encoded = [0u8; 64];
                    let encode_len = corncobs::encode_buf(&msgpack_decoded, &mut cobs_encoded);

                    port.write(&cobs_encoded[..encode_len]).unwrap();

                    println!("Len: {encode_len}, Data: {:?}", &cobs_encoded[..encode_len],)
                }
                None => break,
            }
        }
    });

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
