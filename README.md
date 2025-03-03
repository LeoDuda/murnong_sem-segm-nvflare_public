# Applying Federated Learning in Thermal urban feature semantic segmentation

In this work we applied Federated Learning on a use case of Thermal urban feature semantic segmentation. Unet was used to identify thermal anomalies (hot spots) in urban environments. 
This, to improve the efficiency of energy-related systems.
Due to multiple cities involved and the growing attention in regard of privacy, introducing Federated Learning seemed like a common step.
In particular, [NVFlare](https://github.com/NVIDIA/NVFlare) and its variaty of workflows and features was used.


# How to use
To run the different Federated Workflows either as a simulation or in the real world, the packages shall be installed first:
``` bash
git clone murnong_sem-segm-nvflare
cd  murnong_sem-segm-nvflare
pip install -e .
```

To run the code with the simulator or within a real world using Provisioning, a virtual environment like [venv](https://docs.python.org/3/library/venv.html) is recommended.
The dataset is not contained within this repository but access can be requested.

In this code a private MLFlow instance was used. 
To avoid errors the link should be changed to a local MLFlow instance or an own private server instance within `JOB/app/config/config_fed_server.conf` in the last section of `components` with the id `"mlflow_receiver_with_tracking_uri"`.
For some decentralized workflows which do not include a server, the changes need to be applied within `config_fed_client.conf`.
When using a MLFlow server instance which is protected with a login, the credentials need to be exported within the terminal first. This can be done with:
``` bash
export MLFLOW_TRACKING_USERNAME='username'  
export MLFLOW_TRACKING_PASSWORD='password'  
```
To run the code within a simulation on a local machine, the virtual environment should be startet first. Then run the NVFlare simulator command:

``` bash
source folder/to/venv/bin/activate
nvflare simulator -n 2 -t 2 ./jobs/JOB -w JOB_workspace
```

- `-n`: indicates the amount of sites within the project. This number should align with the number of sites within the config files and within the main code
- `-t`: indicates the amount of threads
- `-w`: indicates the path where the workspace should be created

There are more parameters available which can be looked up in the official repository of NVFlare.
After running the code the folder `JOB_workspace` will be created with different log and config files.

For using the code with multiple real world clients by using Provisioning, please refer to https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html#provisioning

# Structure of this project

```
├── README.md                   <- The top-level README for developers using this project
├── jobs_with_perun             <- Different workflows witht the option of resource tracking available
└── jobs                        <- Folder containing different workflows for without the trackinf of system resources available
    ├── FedAvg                  <- config files and main code for using fedavg as FL algorithm
    ├── FedProx                 <- config files and main code for using fedprox as FL algorithm
    ├── FedOpt                  <- config files and main code for using fedopt as FL algorithm
    ├── SCAFFOLD                <- config files and main code for using scaffold as FL algorithm
    ├── Swarm_Learning          <- config files and main code for using Swarm Learning as a workflow
    ├── Cyclic_Weight_transfer_Centralized         <- config files and main code for using Cyclic Learning with a server for administration involved
    └── Cyclic_Weight_transfer_Centralized              <- config files and main code for using Cyclic Learning without a server as administrator involved
```

# References

Theoretical references:
 - Federated Learning: Collaborative Machine Learning without Centralized Training Data: https://blog.research.google/2017/04/federated-learning-collaborative.html
 - Roth, H. R., et al. (2022). NVIDIA FLARE: Federated Learning from Simulation to Real-World. arXiv. https://arxiv.org/abs/2210.13291
 - H. Brendan McMahan, et al. (2016). Communication-Efficient Learning of Deep Networks from Decentralized Data. arXiv. https://arxiv.org/abs/1602.05629
 - Tian Li, Anit Kumar Sahu, et al. (2020). Federated Optimization in Heterogeneous Networks. arXiv. https://arxiv.org/abs/1812.06127
 - Sashank Reddi, et al. (2021). Adaptive Federated Optimization. arXiv. https://arxiv.org/abs/2003.00295
 - Sai Praneeth Karimireddy, et al. (2021). SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. arXiv. https://arxiv.org/abs/1910.06378

Technical references:
 - NVFlare GitHub Repository:  https://github.com/NVIDIA/NVFlare
 - NVFlare Documentation https://nvflare.readthedocs.io/en/2.3.0/