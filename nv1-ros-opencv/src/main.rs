use core::panic;

use opencv::{
    core::{GpuMat, Point, Scalar, Size, Stream, VecN, Vector},
    cudaarithm, cudaimgproc, highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
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

            let window_front = "opencv front";
            highgui::named_window(&window_front, 0)?;

            let window_rear = "opencv rear";
            highgui::named_window(&window_rear, 0)?;

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
            let mut area_threshold = 2000;
            highgui::create_trackbar(
                "Area Threthold",
                &window_tuner,
                Some(&mut area_threshold),
                10000,
                None,
            )?;

            let cap_front = videoio::VideoCapture::from_file(
                &gstreamer_pipeline(1, 1280, 720, 1280, 720, 60, 2),
                videoio::CAP_GSTREAMER,
            )?;

            let cap_rear = videoio::VideoCapture::from_file(
                &gstreamer_pipeline(0, 1280, 720, 1280, 720, 60, 2),
                videoio::CAP_GSTREAMER,
            )?;

            let mut processor_front = OpenCVProcessor::new(cap_front)?;
            let mut processor_rear = OpenCVProcessor::new(cap_rear)?;

            loop {
                let processor_front_result = processor_front.process(
                    h_min as f64,
                    s_min as f64,
                    v_min as f64,
                    h_max as f64,
                    s_max as f64,
                    v_max as f64,
                    area_threshold,
                )?;
                let processor_rear_result = processor_rear.process(
                    h_min as f64,
                    s_min as f64,
                    v_min as f64,
                    h_max as f64,
                    s_max as f64,
                    v_max as f64,
                    area_threshold,
                )?;

                for rect in processor_front_result {
                    imgproc::rectangle(
                        &mut processor_front.frame_result,
                        rect,
                        Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        8,
                        0,
                    )?;
                }
                for rect in processor_rear_result {
                    imgproc::rectangle(
                        &mut processor_rear.frame_result,
                        rect,
                        Scalar::new(0.0, 255.0, 0.0, 0.0),
                        2,
                        8,
                        0,
                    )?;
                }

                highgui::imshow(&window_front, &processor_front.frame_result)?;
                highgui::imshow(&window_rear, &processor_rear.frame_result)?;

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

pub struct OpenCVProcessor {
    capture: VideoCapture,
    gpu_frame: GpuMat,
    gpu_frame_yuv: GpuMat,
    gpu_frame_yuv_split: Vector<GpuMat>,
    gpu_frame_rgb_clahed: GpuMat,
    gpu_frame_hsv_clahed: GpuMat,
    gpu_frame_masked: GpuMat,
    gpu_frame_tmp: GpuMat,
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
            gpu_frame: GpuMat::new_def()?,
            gpu_frame_yuv: GpuMat::new_def()?,
            gpu_frame_yuv_split: Vector::new(),
            gpu_frame_rgb_clahed: GpuMat::new_def()?,
            gpu_frame_hsv_clahed: GpuMat::new_def()?,
            gpu_frame_masked: GpuMat::new_def()?,
            gpu_frame_tmp: GpuMat::new_def()?,
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
        area_threshold: i32,
    ) -> Result<Vec<opencv::core::Rect>, opencv::Error> {
        let mut frame = Mat::default();
        self.capture.read(&mut frame)?;
        self.gpu_frame.upload(&frame)?;

        let mut stream = Stream::default()?;

        cudaimgproc::cvt_color(
            &self.gpu_frame,
            &mut self.gpu_frame_yuv,
            imgproc::COLOR_RGB2YUV,
            0,
            &mut stream,
        )?;

        let mut clahe = cudaimgproc::create_clahe(2.0, Size::new(8, 8))?;

        cudaarithm::split_1(
            &self.gpu_frame_yuv,
            &mut self.gpu_frame_yuv_split,
            &mut stream,
        )?;

        let mut gpu_frame_front_channel_clahed = GpuMat::new_def()?;
        cudaimgproc::CUDA_CLAHETrait::apply(
            &mut clahe,
            &self.gpu_frame_yuv_split.get(2)?,
            &mut gpu_frame_front_channel_clahed,
            &mut stream,
        )?;

        self.gpu_frame_yuv_split
            .set(2, gpu_frame_front_channel_clahed)?;
        cudaarithm::merge_1(
            &self.gpu_frame_yuv_split,
            &mut self.gpu_frame_yuv,
            &mut stream,
        )?;

        cudaimgproc::cvt_color(
            &self.gpu_frame_yuv,
            &mut self.gpu_frame_rgb_clahed,
            imgproc::COLOR_YUV2RGB,
            0,
            &mut stream,
        )?;

        cudaimgproc::cvt_color(
            &self.gpu_frame_rgb_clahed,
            &mut self.gpu_frame_hsv_clahed,
            imgproc::COLOR_RGB2HSV,
            0,
            &mut stream,
        )?;

        cudaarithm::in_range(
            &self.gpu_frame_hsv_clahed,
            VecN::new(h_min, s_min, v_min, 0.0),
            VecN::new(h_max, s_max, v_max, 0.0),
            &mut self.gpu_frame_masked,
            &mut stream,
        )?;
        cudaarithm::bitwise_not(
            &self.gpu_frame,
            &mut self.gpu_frame_tmp,
            &self.gpu_frame_masked,
            &mut stream,
        )?;

        cudaarithm::bitwise_not(
            &self.gpu_frame_tmp,
            &mut self.gpu_frame_result,
            &self.gpu_frame_masked,
            &mut stream,
        )?;

        stream.wait_for_completion()?;

        self.gpu_frame_result.download(&mut self.frame_result)?;

        self.gpu_frame_masked.download(&mut self.frame_masked)?;

        imgproc::connected_components_with_stats_def(
            &self.frame_masked,
            &mut self.labels,
            &mut self.stats,
            &mut self.centroids,
        )?;

        let mut detected_rects = Vec::new();
        for i in 1..self.stats.rows() {
            let area = self
                .stats
                .at_pt::<i32>(Point::new(imgproc::CC_STAT_AREA, i))?;

            if *area > area_threshold {
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
                detected_rects.push(rect);
            }
        }

        Ok(detected_rects)
    }
}
