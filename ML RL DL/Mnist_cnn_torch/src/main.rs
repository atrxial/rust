use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};
use plotters::prelude::*;

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        Net {
            conv1: nn::conv2d(vs, 1, 32, 3, Default::default()),
            conv2: nn::conv2d(vs, 32, 64, 3, Default::default()),
            fc1: nn::linear(vs, 64 * 5 * 5, 128, Default::default()),
            fc2: nn::linear(vs, 128, 10, Default::default()),
        }
    }
}

impl Module for Net {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 64 * 5 * 5])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

fn main() -> Result<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    let mut accuracies = Vec::new();

    for epoch in 1..=10 {
        train(&net, &mut opt, &m)?;
        let test_accuracy = test_accuracy(&net, &m)?;
        accuracies.push((epoch, test_accuracy));
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy);
    }

    plot_accuracy(&accuracies)?;

    Ok(())
}

fn train(net: &Net, opt: &mut nn::Optimizer, m: &tch::vision::mnist::Mnist) -> Result<()> {
    for (bimages, blabels) in m.train_iter(256).shuffle().to_device(opt.device()) {
        let loss = net
            .forward(&bimages)
            .cross_entropy_for_logits(&blabels);
        opt.backward_step(&loss);
    }
    Ok(())
}

fn test_accuracy(net: &Net, m: &tch::vision::mnist::Mnist) -> Result<f64> {
    let accuracy = net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, net.vs.device(), 1024);
    Ok(accuracy)
}

fn plot_accuracy(accuracies: &[(i32, f64)]) -> Result<()> {
    let root = BitMapBackend::new("accuracy_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Test Accuracy vs Epochs", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..11, 0f64..1f64)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        accuracies.iter().map(|&(x, y)| (x, y)),
        &RED,
    ))?;

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

