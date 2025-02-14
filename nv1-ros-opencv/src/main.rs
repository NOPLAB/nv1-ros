use core::panic;

use opencv::{
    core::{GpuMat, Point, Scalar, Size, Stream, VecN, Vector},
    cudaarithm, cudaimgproc, highgui, imgproc,
    prelude::*,
    videoio,
};
use tokio::{select, task};

fn gstreamer_pipeline(
    sensor_id: u32,
    capture_width: u32,
    capture_height: u32,
    display_width: u32,
    display_height: u32,
    framerate: u32,
    flip_method: u32,
) -> String {
    format!(
        "nvarguscamerasrc sensor-id={} ! video/x-raw(memory:NVMM), width=(int){}, height=(int){}, framerate=(fraction){}/1 ! nvvidconv flip-method={} ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",
        sensor_id,
        capture_width,
        capture_height,
        framerate,
        flip_method,
        display_width,
        display_height
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_opencv", "")?;

    let opencv_handle: tokio::task::JoinHandle<std::result::Result<(), opencv::Error>> =
        task::spawn(async move {
            let window_tuner = "opencv tuner";
            highgui::named_window(window_tuner, 0)?;

            let mut h_min = 0;
            highgui::create_trackbar("H_min", &window_tuner, Some(&mut h_min), 255, None)?;
            let mut h_max = 255;
            highgui::create_trackbar("H_max", &window_tuner, Some(&mut h_max), 255, None)?;
            let mut s_min = 0;
            highgui::create_trackbar("S_min", &window_tuner, Some(&mut s_min), 255, None)?;
            let mut s_max = 255;
            highgui::create_trackbar("S_max", &window_tuner, Some(&mut s_max), 255, None)?;
            let mut v_min = 0;
            highgui::create_trackbar("V_min", &window_tuner, Some(&mut v_min), 255, None)?;
            let mut v_max = 255;
            highgui::create_trackbar("V_max", &window_tuner, Some(&mut v_max), 255, None)?;

            let mut cap_front = videoio::VideoCapture::from_file(
                &gstreamer_pipeline(1, 1280, 720, 1280, 720, 60, 2),
                videoio::CAP_GSTREAMER,
            )?;

            let mut cap_back = videoio::VideoCapture::from_file(
                &gstreamer_pipeline(0, 1280, 720, 1280, 720, 60, 2),
                videoio::CAP_GSTREAMER,
            )?;

            let mut stream = Stream::default()?;

            let mut gpu_frame_front = GpuMat::new_def()?;
            let mut gpu_frame_front_yuv = GpuMat::new_def()?;
            let mut gpu_frame_front_yuv_split = Vector::new();
            let mut gpu_frame_front_rgb_clahed = GpuMat::new_def()?;
            let mut gpu_frame_front_hsv_clahed = GpuMat::new_def()?;
            let mut gpu_mask: GpuMat = GpuMat::new_def()?;
            let mut gpu_tmp: GpuMat = GpuMat::new_def()?;
            let mut dst = Mat::default();
            let mut mask = Mat::default();
            let mut labels = Mat::default();
            let mut stats = Mat::default();
            let mut centroids = Mat::default();

            loop {
                let mut frame_front = Mat::default();
                if cap_front.read(&mut frame_front)? {
                    // let mut frame_front_bluer = Mat::default();
                    // imgproc::gaussian_blur_def(
                    //     &frame_front,
                    //     &mut frame_front_bluer,
                    //     Size::new(5, 5),
                    //     0.0,
                    // )?;

                    gpu_frame_front.upload(&frame_front)?;

                    cudaimgproc::cvt_color(
                        &gpu_frame_front,
                        &mut gpu_frame_front_yuv,
                        imgproc::COLOR_RGB2YUV,
                        0,
                        &mut stream,
                    )?;

                    let mut clahe = cudaimgproc::create_clahe(2.0, Size::new(8, 8))?;

                    cudaarithm::split_1(
                        &gpu_frame_front_yuv,
                        &mut gpu_frame_front_yuv_split,
                        &mut stream,
                    )?;

                    let mut gpu_frame_front_channel_clahed = GpuMat::new_def()?;
                    cudaimgproc::CUDA_CLAHETrait::apply(
                        &mut clahe,
                        &gpu_frame_front_yuv_split.get(2)?,
                        &mut gpu_frame_front_channel_clahed,
                        &mut stream,
                    )?;

                    gpu_frame_front_yuv_split.set(2, gpu_frame_front_channel_clahed)?;
                    cudaarithm::merge_1(
                        &gpu_frame_front_yuv_split,
                        &mut gpu_frame_front_yuv,
                        &mut stream,
                    )?;

                    cudaimgproc::cvt_color(
                        &gpu_frame_front_yuv,
                        &mut gpu_frame_front_rgb_clahed,
                        imgproc::COLOR_YUV2RGB,
                        0,
                        &mut stream,
                    )?;

                    cudaimgproc::cvt_color(
                        &gpu_frame_front_rgb_clahed,
                        &mut gpu_frame_front_hsv_clahed,
                        imgproc::COLOR_RGB2HSV,
                        0,
                        &mut stream,
                    )?;

                    cudaarithm::in_range(
                        &gpu_frame_front_hsv_clahed,
                        VecN::new(h_min as f64, s_min as f64, v_min as f64, 0.0),
                        VecN::new(h_max as f64, s_max as f64, v_max as f64, 0.0),
                        &mut gpu_mask,
                        &mut stream,
                    )?;
                    cudaarithm::bitwise_not(
                        &gpu_frame_front,
                        &mut gpu_tmp,
                        &gpu_mask,
                        &mut stream,
                    )?;

                    let mut gpu_result = GpuMat::new_def()?;
                    cudaarithm::bitwise_not(&gpu_tmp, &mut gpu_result, &gpu_mask, &mut stream)?;

                    stream.wait_for_completion()?;

                    gpu_result.download(&mut dst)?;

                    gpu_mask.download(&mut mask)?;

                    imgproc::connected_components_with_stats_def(
                        &mask,
                        &mut labels,
                        &mut stats,
                        &mut centroids,
                    )?;

                    for i in 1..stats.rows() {
                        let area = stats.at_pt::<i32>(Point::new(imgproc::CC_STAT_AREA, i))?;

                        if *area > 2000 {
                            let left = stats.at_pt::<i32>(Point::new(imgproc::CC_STAT_LEFT, i))?;
                            let top = stats.at_pt::<i32>(Point::new(imgproc::CC_STAT_TOP, i))?;
                            let width =
                                stats.at_pt::<i32>(Point::new(imgproc::CC_STAT_WIDTH, i))?;
                            let height =
                                stats.at_pt::<i32>(Point::new(imgproc::CC_STAT_HEIGHT, i))?;
                            let rect = opencv::core::Rect::new(*left, *top, *width, *height);

                            imgproc::rectangle(
                                &mut dst,
                                rect,
                                Scalar::new(0.0, 255.0, 0.0, 0.0),
                                2,
                                8,
                                0,
                            )?;
                        }
                    }

                    // highgui::imshow(window_camera, &dst)?;
                    highgui::imshow(&window_tuner, &dst)?;
                }

                let key = opencv::highgui::wait_key(1)?;
                if key == 27 {
                    break;
                }
            }

            Ok(())
        });

    let ros_handle = tokio::spawn(async move {
        loop {
            node.spin_once(std::time::Duration::from_millis(100));
        }
    });

    select! {
        res = opencv_handle => res?,
        res = ros_handle => res?,
    }
    .unwrap();

    Ok(())
}
