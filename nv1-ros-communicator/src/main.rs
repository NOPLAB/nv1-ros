use futures::stream::StreamExt;
use r2r::QosProfile;

use std::time::Duration;
use tokio::task;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Serial {
    x: f32,
    y: f32,
    angle: f32,
    kick: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
union SerialData {
    serial: Serial,
    buffer: [u8; size_of::<Serial>()],
}

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

        let mut port = serialport::new("/dev/ttyTCU0", 115200)
            .timeout(Duration::from_millis(10))
            .open()
            .expect("Failed to open port");

        loop {
            match sub_teleop.next().await {
                Some(msg) => {
                    println!("{:#?}", msg);

                    let serial = Serial {
                        x: msg.linear.x as f32,
                        y: msg.linear.z as f32,
                        angle: msg.angular.z as f32,
                        kick: false,
                    };

                    let encode = SerialData { serial: serial };

                    let mut cobs_encoded = [0u8; 64];
                    let encode_len =
                        corncobs::encode_buf(&unsafe { encode.buffer }, &mut cobs_encoded);

                    port.write(&cobs_encoded[..encode_len]).unwrap();

                    println!("{:?}", &cobs_encoded[..encode_len])
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
