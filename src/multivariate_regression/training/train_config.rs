use crate::multivariate_regression::cost_fn::CostFn;
use crate::multivariate_regression::gradient::Gradient;
use crate::multivariate_regression::normalization::NormalizationParameterType;
use crate::multivariate_regression::regularization::Regularization;
use crate::multivariate_regression::update_weight::{MiniBatchSize, UpdatationMethod};

pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    pub normalization_parameter_type: Option<NormalizationParameterType>,
    pub optimizer: Option<UpdatationMethod>,
    pub mini_batch_size: Option<MiniBatchSize>,
    pub regularization: Option<Regularization>,
    pub cost_fn: Option<CostFn>,
    pub gradient_fn: Option<Gradient>,
    pub delta: Option<f64>,
    pub print_log: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            lr: 0.01,
            normalization_parameter_type: None,
            optimizer: None,
            mini_batch_size: None,
            regularization: None,
            cost_fn: None,
            gradient_fn: None,
            delta: Some(1.0),
            print_log: false,
        }
    }
}

pub struct TrainConfigBuilder {
    config: TrainConfig,
}

impl TrainConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: TrainConfig::default(),
        }
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.lr = lr;
        self
    }

    pub fn normalization(mut self, normalization: NormalizationParameterType) -> Self {
        self.config.normalization_parameter_type = Some(normalization);
        self
    }

    pub fn optimizer(mut self, optimizer: UpdatationMethod) -> Self {
        self.config.optimizer = Some(optimizer);
        self
    }

    pub fn mini_batch_size(mut self, size: MiniBatchSize) -> Self {
        self.config.mini_batch_size = Some(size);
        self
    }

    pub fn regularization(mut self, reg: Regularization) -> Self {
        self.config.regularization = Some(reg);
        self
    }

    pub fn cost_fn(mut self, cost: CostFn) -> Self {
        self.config.cost_fn = Some(cost);
        self
    }

    pub fn gradient_fn(mut self, gradient: Gradient) -> Self {
        self.config.gradient_fn = Some(gradient);
        self
    }

    pub fn delta(mut self, delta: f64) -> Self {
        self.config.delta = Some(delta);
        self
    }

    pub fn print_log(mut self, print_log: bool) -> Self {
        self.config.print_log = print_log;
        self
    }

    pub fn build(self) -> TrainConfig {
        self.config
    }
}