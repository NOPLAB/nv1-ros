use std::sync::Arc;

use clap::Parser;
use futures::StreamExt;
use opencv::{
    core::{GpuMat, Point, Scalar, Size, Stream, VecN, Vector},
    cudaarithm, cudaimgproc, highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
};
use r2r::QosProfile;
use tokio::sync::Mutex;
use tokio::task;

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

fn convert_pixet_to_theta(x: i32) -> f64 {
    (x - 360) as f64 / 6.0
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = false)]
    display: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "nv1_ros_opencv", "")?;

    let pub_nv1_opencv_own = node.create_publisher::<r2r::std_msgs::msg::Float64>(
        "/nv1/opencv/own",
        QosProfile::sensor_data(),
    )?;
    let pub_nv1_opencv_opp = node.create_publisher::<r2r::std_msgs::msg::Float64>(
        "/nv1/opencv/opp",
        QosProfile::sensor_data(),
    )?;

    let mut sub_nv1_opencv_hsv_own = node.subscribe::<r2r::std_msgs::msg::UInt8MultiArray>(
        "/nv1/opencv/hsv_own",
        QosProfile::default(),
    )?;
    let mut sub_nv1_opencv_hsv_opp = node.subscribe::<r2r::std_msgs::msg::UInt8MultiArray>(
        "/nv1/opencv/hsv_opp",
        QosProfile::default(),
    )?;

    let hsv_own = Arc::new(Mutex::new([0u8; 6]));
    let hsv_opp = Arc::new(Mutex::new([0u8; 6]));

    let hsv_own_c = hsv_own.clone();
    task::spawn(async move {
        loop {
            let msg = sub_nv1_opencv_hsv_own.next().await;

            if msg.is_none() {
                continue;
            }

            let msg = msg.unwrap();

            if msg.data.len() != 6 {
                continue;
            }

            let mut hsv_own = hsv_own_c.lock().await;
            hsv_own.copy_from_slice(&msg.data);
        }
    });

    let hsv_opp_c = hsv_opp.clone();
    task::spawn(async move {
        loop {
            let msg = sub_nv1_opencv_hsv_opp.next().await;

            if msg.is_none() {
                continue;
            }

            let msg = msg.unwrap();

            if msg.data.len() != 6 {
                continue;
            }

            let mut hsv_opp = hsv_opp_c.lock().await;
            hsv_opp.copy_from_slice(&msg.data);
        }
    });

    task::spawn(async move {
        let window_tuner = "opencv tuner";
        let window_front = "opencv front";
        let window_rear = "opencv rear";

        if args.display {
            highgui::named_window(window_tuner, 0).unwrap();
            highgui::named_window(&window_front, 0).unwrap();
            highgui::named_window(&window_rear, 0).unwrap();

            // highgui::create_trackbar("H_min", &window_tuner, Some(&mut h_min), 255, None)?;
            // highgui::create_trackbar("H_max", &window_tuner, Some(&mut h_max), 255, None)?;
            // highgui::create_trackbar("S_min", &window_tuner, Some(&mut s_min), 255, None)?;
            // highgui::create_trackbar("S_max", &window_tuner, Some(&mut s_max), 255, None)?;
            // highgui::create_trackbar("V_min", &window_tuner, Some(&mut v_min), 255, None)?;
            // highgui::create_trackbar("V_max", &window_tuner, Some(&mut v_max), 255, None)?;
        }

        let cap_front = videoio::VideoCapture::from_file(
            &gstreamer_pipeline(1, 720, 480, 720, 480, 60, 2),
            videoio::CAP_GSTREAMER,
        )
        .unwrap();

        let _cap_rear = videoio::VideoCapture::from_file(
            &gstreamer_pipeline(0, 720, 480, 720, 480, 60, 2),
            videoio::CAP_GSTREAMER,
        )
        .unwrap();

        let mut processor_front = OpenCVProcessor::new(cap_front).unwrap();
        // let mut processor_rear = OpenCVProcessor::new(cap_rear)?;

        let mut goal_theta_front = 0.0;

        loop {
            let hsv_opp = hsv_opp.lock().await.clone();
            let processor_front_result = processor_front
                .process(
                    hsv_opp[0] as f64,
                    hsv_opp[1] as f64,
                    hsv_opp[2] as f64,
                    hsv_opp[3] as f64,
                    hsv_opp[4] as f64,
                    hsv_opp[5] as f64,
                )
                .unwrap();

            if let Some(rect) = processor_front_result {
                imgproc::rectangle(
                    &mut processor_front.frame_result,
                    rect,
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    8,
                    0,
                )
                .unwrap();

                let center_x = rect.x + rect.width / 2;

                goal_theta_front = convert_pixet_to_theta(center_x);
            }

            let _ = pub_nv1_opencv_opp.publish(&r2r::std_msgs::msg::Float64 {
                data: goal_theta_front,
            });

            if args.display {
                highgui::imshow(&window_front, &processor_front.frame_result).unwrap();

                let key = opencv::highgui::wait_key(1).unwrap();
                if key == 27 {
                    break;
                }
            }
        }
    });

    let handle = std::thread::spawn(move || loop {
        node.spin_once(std::time::Duration::from_millis(100));
    });
    handle.join().unwrap();

    Ok(())
}

pub struct OpenCVProcessor {
    capture: VideoCapture,
    stream: Stream,
    gpu_frame: GpuMat,
    gpu_frame_yuv: GpuMat,
    gpu_frame_yuv_split: Vector<GpuMat>,
    gpu_frame_rgb_clahed: GpuMat,
    gpu_frame_hsv_clahed: GpuMat,
    gpu_frame_masked: GpuMat,
    gpu_frame_result: GpuMat,
    pub frame_result: Mat,
    pub frame_masked: Mat,
    pub labels: Mat,
    pub stats: Mat,
    pub centroids: Mat,
}

impl OpenCVProcessor {
    pub fn new(capture: VideoCapture) -> Result<Self, opencv::Error> {
        Ok(OpenCVProcessor {
            capture,
            stream: Stream::default()?,
            gpu_frame: GpuMat::new_def()?,
            gpu_frame_yuv: GpuMat::new_def()?,
            gpu_frame_yuv_split: Vector::new(),
            gpu_frame_rgb_clahed: GpuMat::new_def()?,
            gpu_frame_hsv_clahed: GpuMat::new_def()?,
            gpu_frame_masked: GpuMat::new_def()?,
            gpu_frame_result: GpuMat::new_def()?,
            frame_result: Mat::default(),
            frame_masked: Mat::default(),
            labels: Mat::default(),
            stats: Mat::default(),
            centroids: Mat::default(),
        })
    }

    pub fn process(
        &mut self,
        h_min: f64,
        s_min: f64,
        v_min: f64,
        h_max: f64,
        s_max: f64,
        v_max: f64,
    ) -> Result<Option<opencv::core::Rect>, opencv::Error> {
        let mut frame = Mat::default();
        self.capture.read(&mut frame)?;
        self.gpu_frame.upload(&frame)?;

        cudaimgproc::cvt_color(
            &self.gpu_frame,
            &mut self.gpu_frame_yuv,
            imgproc::COLOR_RGB2YUV,
            0,
            &mut self.stream,
        )?;

        let mut clahe = cudaimgproc::create_clahe(2.0, Size::new(8, 8))?;

        cudaarithm::split_1(
            &self.gpu_frame_yuv,
            &mut self.gpu_frame_yuv_split,
            &mut self.stream,
        )?;

        let mut gpu_frame_front_channel_clahed = GpuMat::new_def()?;
        cudaimgproc::CUDA_CLAHETrait::apply(
            &mut clahe,
            &self.gpu_frame_yuv_split.get(2)?,
            &mut gpu_frame_front_channel_clahed,
            &mut self.stream,
        )?;

        self.gpu_frame_yuv_split
            .set(2, gpu_frame_front_channel_clahed)?;
        cudaarithm::merge_1(
            &self.gpu_frame_yuv_split,
            &mut self.gpu_frame_yuv,
            &mut self.stream,
        )?;

        cudaimgproc::cvt_color(
            &self.gpu_frame_yuv,
            &mut self.gpu_frame_rgb_clahed,
            imgproc::COLOR_YUV2RGB,
            0,
            &mut self.stream,
        )?;

        cudaimgproc::cvt_color(
            &self.gpu_frame_rgb_clahed,
            &mut self.gpu_frame_hsv_clahed,
            imgproc::COLOR_RGB2HSV,
            0,
            &mut self.stream,
        )?;

        cudaarithm::in_range(
            &self.gpu_frame_hsv_clahed,
            VecN::new(h_min, s_min, v_min, 0.0),
            VecN::new(h_max, s_max, v_max, 0.0),
            &mut self.gpu_frame_masked,
            &mut self.stream,
        )?;

        self.gpu_frame_result = GpuMat::new_rows_cols_with_default_def(
            self.gpu_frame.rows(),
            self.gpu_frame.cols(),
            self.gpu_frame.typ()?,
            0.into(),
        )?;

        cudaarithm::bitwise_not(
            &self.gpu_frame,
            &mut self.gpu_frame_result,
            &self.gpu_frame_masked,
            &mut self.stream,
        )?;

        cudaarithm::bitwise_not(
            &self.gpu_frame_result.clone(),
            &mut self.gpu_frame_result,
            &self.gpu_frame_masked,
            &mut self.stream,
        )?;

        self.stream.wait_for_completion()?;

        self.gpu_frame_result.download(&mut self.frame_result)?;
        self.gpu_frame_masked.download(&mut self.frame_masked)?;

        imgproc::connected_components_with_stats_def(
            &self.frame_masked,
            &mut self.labels,
            &mut self.stats,
            &mut self.centroids,
        )?;

        let mut detected_rect = None;
        let mut max_area = 0;
        for i in 1..self.stats.rows() {
            let area = self
                .stats
                .at_pt::<i32>(Point::new(imgproc::CC_STAT_AREA, i))?;

            if *area > max_area {
                let left = self
                    .stats
                    .at_pt::<i32>(Point::new(imgproc::CC_STAT_LEFT, i))?;
                let top = self
                    .stats
                    .at_pt::<i32>(Point::new(imgproc::CC_STAT_TOP, i))?;
                let width = self
                    .stats
                    .at_pt::<i32>(Point::new(imgproc::CC_STAT_WIDTH, i))?;
                let height = self
                    .stats
                    .at_pt::<i32>(Point::new(imgproc::CC_STAT_HEIGHT, i))?;
                let rect = opencv::core::Rect::new(*left, *top, *width, *height);

                max_area = *area;
                detected_rect = Some(rect);
            }
        }

        Ok(detected_rect)
    }
}
