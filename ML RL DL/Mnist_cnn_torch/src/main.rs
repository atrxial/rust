use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

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

    for epoch in 1..=10 {
        for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
            let loss = net
                .forward(&bimages)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy = net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    Ok(())
}