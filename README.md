## Learning Physics-Informed Neural Networks without Stacked Back-propagation

This is the official implementation of [*Learning Physics-Informed Neural Networks without Stacked Back-propagation*](https://arxiv.org/abs/2202.09340) (**AISTATS 2023**).

The required package can be found in `requirements.txt`. To install the required package, run `pip install -r requirements.txt`.

To reproduce the result of our model on Possion's Equation, run `python run.py task=possion`.

To reproduce the result of our model on Heat's Equation, run `python run.py task=heat`.

To reproduce the result of our model on HJB Equation, run `python run.py task=hjb`.

To reproduce the PINN baselines, you can edit `conf/config.yaml` to set PINN as the default setting.