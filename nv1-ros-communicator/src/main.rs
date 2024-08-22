use futures::{lock::Mutex, stream::StreamExt};
use r2r::QosProfile;

use std::{sync::Arc, time::Duration};
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_communicator", "")?;

    let mut sub_teleop =
        node.subscribe::<r2r::geometry_msgs::msg::Twist>("/cmd_vel", QosProfile::default())?;

    let ports = serialport::available_ports().expect("No ports found!");
    for p in ports {
        println!("{}", p.port_name);
    }

    let port = Arc::new(Mutex::new(
        serialport::new("/dev/ttyTHS1", 115200)
            .timeout(Duration::from_millis(5))
            .open()
            .expect("Failed to open port"),
    ));

    let port_task = port.clone();
    task::spawn(async move {
        loop {
            let cmd_vel = sub_teleop.next().await;
            match cmd_vel {
                Some(cmd_vel) => {
                    println!("{:#?}", cmd_vel);

                    let send_msg = nv1_msg::hub::HubMsgPackRx {
                        vel: nv1_msg::hub::Velocity {
                            x: cmd_vel.linear.z as f32,
                            y: cmd_vel.linear.x as f32,
                            angle: cmd_vel.angular.z as f32,
                        },
                        kick: false,
                    };

                    let msg_cobs = postcard::to_stdvec_cobs(&send_msg).unwrap();

                    port_task.lock().await.write(&msg_cobs).unwrap();

                    println!(
                        "[UART TX] send Len: {}, Data: {:?}",
                        msg_cobs.len(),
                        msg_cobs
                    );

                    // let mut msg_decode_test = msg_cobs.clone();
                    // let msg_decoded = postcard::from_bytes_cobs::<nv1_msg::hub::HubMsgPackRx>(
                    //     &mut msg_decode_test,
                    // )
                    // .unwrap();
                    // println!("{:#?}", msg_decoded);
                }
                None => break,
            }
        }
    });

    let port_task = port.clone();
    task::spawn(async move {
        const HUB_MSG_TX_SIZE: usize = 42;

        loop {
            let mut buf = [0; HUB_MSG_TX_SIZE];
            let res = port_task.lock().await.read(&mut buf);
            match res {
                Ok(n) => {
                    if n != HUB_MSG_TX_SIZE {
                        println!("Read error: {}", n);
                        continue;
                    }

                    match postcard::from_bytes_cobs::<nv1_msg::hub::HubMsgPackTx>(&mut buf) {
                        Ok(msg) => {
                            println!("{:#?}", msg);

                            if msg.shutdown {
                                println!("System Shutdown...");
                                system_shutdown::shutdown().unwrap();
                            }

                            if msg.reboot {
                                println!("System Reboot...");
                                system_shutdown::reboot().unwrap();
                            }
                        }
                        Err(_) => {
                            println!("Decode error");
                        }
                    };
                }
                Err(err) => {
                    // println!("Read error: {}", err);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
            };
        }
    });

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
