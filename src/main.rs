use candle::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam;
use candle::Tensor;

fn main() -> anyhow::Result<()> {
    let device = device(true)?;
    let (image, initial_h, initial_w) = load_image("bike.jpg", Some(sam::IMAGE_SIZE))?;
    let image = image.to_device(&device)?;
    println!("----- {:?}", image);


    let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/candle-sam".to_string());
            let filename = if true {
                "mobile_sam-tiny-vitt.safetensors"
            } else {
                "sam_vit_b_01ec64.safetensors"
            };
    let model = api.get(filename)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model], candle::DType::F32, &device)?
    };
    let sam = sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)?; // sam_vit_b
    let points: Vec<String> = vec![];
    let iter_points = points.iter().map(|p| (p, true));
    let neg_points: Vec<String> = vec![];
    let iter_neg_points = neg_points.iter().map(|p| (p, false));
    let points = iter_points
        .chain(iter_neg_points)
        .map(|(point, b)| {
            use std::str::FromStr;
            let xy = point.split(',').collect::<Vec<_>>();
            if xy.len() != 2 {
                anyhow::bail!("expected format for points is 0.4,0.2")
            }
            Ok((f64::from_str(xy[0])?, f64::from_str(xy[1])?, b))
        }) 
        .collect::<anyhow::Result<Vec<_>>>()?;
    let start_time = std::time::Instant::now();
    let (mask, iou_preductions) = sam.forward(&image, &points, false)?;

    println!("mask:\n {}", mask);
    println!("iou_predictions: {}", iou_preductions);
    println!("mask generated in {:.2}s", start_time.elapsed().as_secs_f32());

    Ok(())
}


pub fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(device)
    }
}

pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>
) -> anyhow::Result<(Tensor, usize, usize)> {
    let img = image::io::Reader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?;

    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    Ok((data, initial_h, initial_w))
}

