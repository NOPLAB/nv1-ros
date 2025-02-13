use futures::{lock::Mutex, stream::StreamExt};
use r2r::QosProfile;

use std::{cell::RefCell, sync::Arc, time::Duration};
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_communicator", "")?;

    let mut sub_teleop =
        node.subscribe::<r2r::geometry_msgs::msg::Twist>("/cmd_vel", QosProfile::sensor_data())?;

    unsafe {
        let status = jetgpio_sys::gpioInitialise();
        println!("gpioInitialise {}", status);
        let status = jetgpio_sys::gpioSetMode(32, jetgpio_sys::JET_OUTPUT);
        println!("gpioSetMode {}", status);
    }

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

    let have_ball = Arc::new(Mutex::new(RefCell::new(false)));
    let port_task = port.clone();
    task::spawn(async move {
        loop {
            let cmd_vel = sub_teleop.next().await;
            match cmd_vel {
                Some(cmd_vel) => {
                    // println!("{:#?}", cmd_vel);

                    let send_msg = nv1_msg::hub::HubMsgPackRx {
                        vel: nv1_msg::hub::Velocity {
                            x: cmd_vel.linear.x as f32,
                            y: cmd_vel.linear.y as f32,
                            angle: cmd_vel.angular.z as f32,
                        },
                        kick: false, // dont use
                    };

                    let msg_cobs = postcard::to_stdvec_cobs(&send_msg).unwrap();

                    port_task.lock().await.write(&msg_cobs).unwrap();
                }
                None => break,
            }
        }
    });

    let pub_speed = node.create_publisher::<r2r::geometry_msgs::msg::Vector3>(
        "/nv1/speed",
        QosProfile::sensor_data(),
    )?;
    let pub_ir = node.create_publisher::<r2r::geometry_msgs::msg::Vector3>(
        "/nv1/ir",
        QosProfile::sensor_data(),
    )?;
    let pub_have_ball = node.create_publisher::<r2r::std_msgs::msg::Bool>(
        "/nv1/have_ball",
        QosProfile::sensor_data(),
    )?;

    let have_ball_task = have_ball.clone();
    let port_task = port.clone();
    task::spawn(async move {
        const HUB_MSG_TX_SIZE: usize = 42;
        let mut prev_kick = false;

        loop {
            let buffer_len = port_task.lock().await.bytes_to_read().unwrap();
            if buffer_len < HUB_MSG_TX_SIZE as u32 {
                // println!("Buffer len: {}", buffer_len);
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            } else if buffer_len > HUB_MSG_TX_SIZE as u32 {
                let mut buf = [0; 1];
                loop {
                    port_task.lock().await.read(&mut buf).unwrap();
                    if buf == [0] {
                        break;
                    }
                }
                continue;
            }

            let mut buf = [0; HUB_MSG_TX_SIZE];
            let res = port_task.lock().await.read(&mut buf);
            match res {
                Ok(n) => {
                    if n != HUB_MSG_TX_SIZE {
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

                            let pub_msg_speed = r2r::geometry_msgs::msg::Vector3 {
                                x: msg.vel.x as f64,
                                y: msg.vel.y as f64,
                                z: msg.vel.angle as f64,
                            };

                            let ir_strength = if msg.ir.strength > 2.0 {
                                0.0
                            } else {
                                msg.ir.strength as f64
                            };

                            let pub_msg_ir = r2r::geometry_msgs::msg::Vector3 {
                                x: msg.ir.x as f64,
                                y: msg.ir.y as f64,
                                z: ir_strength,
                            };

                            have_ball_task.lock().await.replace(msg.have_ball);

                            let pub_msg_have_ball = r2r::std_msgs::msg::Bool {
                                data: msg.have_ball,
                            };

                            if msg.have_ball && msg.vel.angle.abs() < 2.0 {
                                if !prev_kick {
                                    unsafe {
                                        let _status = jetgpio_sys::gpioWrite(32, 1);
                                    }

                                    tokio::time::sleep(Duration::from_millis(100)).await;

                                    unsafe {
                                        let _status = jetgpio_sys::gpioWrite(32, 0);
                                    }
                                }

                                prev_kick = true;
                            } else {
                                prev_kick = false;
                            }

                            pub_speed.publish(&pub_msg_speed).unwrap();
                            pub_ir.publish(&pub_msg_ir).unwrap();
                            pub_have_ball.publish(&pub_msg_have_ball).unwrap();
                        }
                        Err(_) => {
                            println!("Decode error");
                        }
                    };
                }
                Err(_) => {
                    continue;
                }
            };
        }
    });

    // let mut sub_kicker =
    //     node.subscribe::<r2r::std_msgs::msg::Bool>("/nv1/kicker", QosProfile::sensor_data())?;

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}
