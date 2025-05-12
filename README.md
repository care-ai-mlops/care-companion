# Care Companion: Comprehensive AI-Powered Medical Imaging and Patient Care Solution

## Value Proposition
The  Care Companion system will be seamlessly integrated into existing healthcare practices, providing doctors with enhanced diagnostic tools. In traditional healthcare systems, radiologists face challenges analyzing numerous X-rays and CT scans, leading to delays, burnout, and missed diagnoses. By incorporating AI-powered image analysis, doctors can quickly identify fractures from X-rays, and detect signs of pneumonia and tuberculosis in chest X-rays, improving diagnostic accuracy. This AI-powered system will save time for healthcare professionals, enabling them to focus on patient care. It will enhance workflow in hospitals, clinics, and radiology centers, allowing healthcare providers to deliver faster, more reliable results with Real-time predictions and continuous feedback. 

The system's success will be evaluated based on its ability to enhance diagnostic accuracy, reduce operational costs, and improve patient satisfaction. However, in medical AI, where human lives are at stake, performance metrics prioritize patient safety over raw accuracy. While precision and AUC-ROC are tracked, we focus on optimizing the model for high recall (sensitivity) to minimize false negatives—ensuring no critical fracture, pneumonia, or TB case goes undetected, even at the cost of more false positives. This aligns with clinical best practices, where follow-up tests for potential false alarms are preferable to missed diagnoses. By balancing recall with clinician trust, we reduce liability risks while enabling early interventions that cut long-term costs.

While the AI-powered detection system enhances diagnostic accuracy, it should not be solely relied upon for final decisions. The predictions made by the system are assistance tools, and healthcare professionals should always validate the results through their expertise and further testing. We also employ a warning system that provides the confidence for each prediction made while also alerting the radiologist  that the predictions made can be incorrect and needs a thorough review before finalizing the report. Additionally, we create a dashboard that allows the technicians to monitor the data drift for any potential flaws in the system. 

## Contributors
| Name                   | Responsible For | Link to their commits in this repo                                                       |
|------------------------|-----------------|------------------------------------------------------------------------------------------|
| Satyam Chatrola         | Model Serving and Monitoring |[Commits](https://github.com/care-ai-mlops/care-companion/commits/main/?author=Nightshade14) |
| Akash Peddaputha        | Data Pipeline |[Commits](https://github.com/care-ai-mlops/care-companion/commits/main/?author=akashp04)|
| Yash Patel              | Continuous X Pipeline |[Commits](https://github.com/care-ai-mlops/care-companion/commits/main/?author=YashPatel166)|
| Mamidala Sai Sandeep    | Model Training |[Commits](https://github.com/care-ai-mlops/care-companion/commits/main/?author=Sandeep2229)|

## System Design
![Image](images/systemdesign.png)


## Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used.-->

| Entity | How it was created | Conditions of use |
| :-- | :-- | :-- |
| **Dataset - 1: Wrist X-Ray images** | The dataset is sourced from Kaggle. It contains X-Ray images of wrists. The class distribution of X-rays is equal between fractured and normal non-fractured wrists. The dataset has X-ray images in PA, oblique and lateral projections and for left and right wrists making it robust and covers all major cases. The dataset's balanced nature will mitigate potential bias in model training, while its variety of projection angles will enhance the model's generalizability to real-world clinical scenarios where X-rays may be captured from different orientations. The dataset consists of 12.7K images which totals in size of 7.82 GB. The dataset link is as follows: https://www.kaggle.com/datasets/sirajbunery/wrist-xray-dataset-balanced-data | Open-source<br><br>To train fracture detection models. |
| **Dataset - 2: Chest X-Ray images** | Data Sources<br>OCT Dataset (Kermany): 4,273 pneumonia-infected and 1,583 normal X-rays.<br>RSNA CXR Dataset: 3,500 TB-positive X-rays.<br>NIAID TB Dataset: 3,499 TB-positive X-rays from seven countries.<br>NLM Dataset:<br>Montgomery: 138 X-rays (57 TB, 80 normal).<br>Shenzhen: 662 X-rays (336 TB, 326 normal).<br>Belarus Dataset: 304 TB-infected X-rays (512×512 resolution)<br>Non-X-ray Dataset: 1,357 images of objects (e.g., animals, bikes) from Pavan Sanagapati.<br>Sources<br>Kermany DS, Goldbaum M, Cai W, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell. 2018;172(5):1122-1131.e9. doi:10.1016/j.cell.2018.02.010<br>Radiological Society of North America. Published August 27,2018.Accessed December 10.2024. https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge<br>National Institute of Allergy and Infectious Diseases (NIAID). NIAID TB Portals Dataset. Accessed February 10, 2024. Available at: https://tbportals.niaid.nih.gov/download-data<br>Jaeger S, Candemir S, Antani S, Wáng YX, Lu PX, Thoma G. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. Quant Imaging Med Surg. 2014;4(6):475-477. doi:10.3978/j.issn.2223-4292.2014.11.20<br>Gabrielian A, Engle E, Harris M, et al. Comparative analysis of genomic variability for drug-resistant strains of Mycobacterium tuberculosis: The special case of Belarus. Infect Genet Evol. 2020;78:104137. doi:10.1016/j.meegid.2019.104137<br>Sanagapati P, Images Dataset. Kaggle. Published 7 years ago. Accessed December3,2024. https://www.kaggle.com/datasets/pavansanagapati/images-dataset/data<br><br>The dataset has 15.3K images and is of size 7.02 GB.<br><br>Dataset link: https://www.kaggle.com/datasets/rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis | Open-source<br><br>To train Pneumonia and Tuberculosis detection models. |
| **Model - 1: CNN** | We plan to experiment with multiple pre-trained CNN backbone architectures like DenseNet-121, InceptionV3/V2, MobileNet etc for fracture detection. The transfer learning approach is the most efficient way of training the models for our use-case. | Used to detect bone fracture in the X-Ray images uploaded by Radiology technician |
| **Model - 2: CNN** | Similarly for Tuberculosis or Pneumonia detection, we plan to use Transfer Learning technique and fine-tune pre-trained models to make predictions. | Used to detect Tuberculosis and Pneumonia in the X-Ray images uploaded by Radiology technician |
| **Model-3: LLM** | We plan to use a self-hosted open-source LLM such as Llama for the notes generation part and/or patient care review sentiment classification. If we use Llama, it reduces the work of fine-tuning as they might have already been trained on such data. | Open-source<br><br>Used to generate patient friendly clinical notes and/or patient care review sentiment classification. |

`Difficulty Points` Attempting from Unit-1:
<ul> <li>Using Multiple Models</li> </ul>

### Summary of Infrastructure Requirements
| **Requirement**      | **How many / When**                      | **Justification**                                                                                                                                     |
|----------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `m1.medium`          | 1 instance, running 24/7                | Used for **backend tasks** such as **PostgreSQL database**, **logging**, and light processing that doesn't require heavy GPU acceleration. This is a lightweight node to handle non-GPU tasks. |
| `gpu_v100`           | 4-hour block, twice a week               | **High-performance model training** for deep learning tasks (e.g., CNN, Vision Transformers). The **V100** provides excellent compute power for training large models on complex datasets. |
| `gpu_a30`            | 4-hour block, 3-4 times a week for inference | **Low-latency inference** for real-time AI predictions (fracture detection, pneumonia, TB). The **A30** provides efficient inference for production with cost-effectiveness. It can handle large models and datasets. |
| **Floating IPs**     | 1-2 for public access to inference model | To provide access to the **AI-powered inference service** (e.g., API for predictions) via a **public-facing IP** for external users (doctors, hospitals, etc.). |
| **Persistent Storage**| 1TB SSD (for model, dataset, and logs)  | To store models, large datasets, and logs used in training and inference. The SSD ensures **fast read/write** speeds necessary for processing large medical images. |


### Detailed Design Plan

#### Model training
##### Strategy and Justification
###### Model-1 (Fracture Detection CNN):
`Model Choice`: We will start with pre-trained CNN backbones like DenseNet-121 (highest AUC on MURA dataset), with comparative testing of MobileNet (latency-optimized) and InceptionV3 (spatial feature strength) and fine-tune them on our fracture detection dataset. We will evaluate multiple architectures to identify the one that balances accuracy, latency, and model size.<br/>
`Training`: Fine-tune on MURA (40k X-rays) via transfer learning. Retrain weekly with 500–1k radiologist corrections using Ray Train on gpu_v100.

###### Model-2 (TB/Pneumonia CNN):
`Model Choice`: Similarly, for TB or pneumonia detection, we will fine-tune pre-trained models using the same approach but with a different dataset.<br/>
`Training`: Initialize with NIH ChestX-ray14 weights. 

###### Model-3 (LLM for Notes):
`Model Choice`: We will use a Llama-2-7B (open-source, locally deployed) fine-tuned with LoRA to generate patient-friendly notes. This model will require minimal fine-tuning, as it has already been trained on large amounts of textual data. <br/>
`Training`: Generate 5k synthetic reports (radiology notes ↔ patient summaries) for fine-tuning

`CNNs are highly suitable for image classification tasks because they are designed to automatically learn spatial hierarchies in data. This makes them especially effective for tasks like detecting fractures or identifying abnormalities in X-ray or CT images.` 
<br>
`Using an LLM like Llama allows us to generate human-readable summaries from complex medical terminology, making the information accessible to patients. Llama and other open-source LLMs come pretrained on vast amounts of textual data, reducing the amount of fine-tuning required for the specific task of converting technical notes to layman's terms.`

###### Workflow Integration:
`Feedback Loop`: Corrections from doctors update PostgreSQL → trigger retraining for CNNs. <br>
`Deployment`: CI/CD (GitHub Actions) promotes models to gpu_a30 (CNNs) and m1.medium (LLM).


#### Training Platforms

We will use MLflow for tracking our model training experiments on Chameleon Cloud. MLflow will serve as our centralized experiment tracking server, where we will log all experiment details, including hyperparameters, model architecture, metrics, and performance. 
 During training, we will log key details such as:
<ul> 
 <li>Hyperparameters (learning rate, batch size, model architecture)</li>
 <li>Performance metrics (accuracy, precision, recall, F1-score)</li>
 <li>Training and validation loss</li>
 <li>Model checkpoints (for retraining)</li>  
</ul>

To efficiently manage and scale our training jobs, we will use a Ray cluster
<ul><li>Parallelized training of models like DenseNet-121, InceptionV3, and MobileNet for fracture detection and TB/pneumonia detection. </li>
  <li>Hyperparameter tuning using Ray Tune, enabling us to experiment with different hyperparameter configurations to find the optimal settings for each model.</li>
</ul>

`Difficulty Points` Attempting from Unit-4 & Unit-5:
<ul> <li>Using ray Train </li> </ul>

#### Model serving 
##### API Endpoint: 
We wrapped our models in a FastAPI endpoint for real-time inference (pneumonia, TB or normal). FastAPI is chosen for its speed and async capabilities, which are crucial for handling real-time inference requests with minimal latency (1-2 seconds per image). 
<ul>
 <li>Size: Models will range from 100MB to 200MB (CNNs for image detection).</li>
 <li>Throughput: 50-100 images per minute for batch inference.</li>
 <li>Latency: less than 1 second on both gpu and cpu.</li>
 <li>Concurrency: Support 75+ concurrent requests for multiple users (doctors)</li>
</ul>

We are using Nvidia Triton server with onnx runtime gpu serving and OpenVINO cpu serving both from triton server with dynamic batching.

The api endpoint has metrics, health and prediction endpoints.

For serving any chameleon node is suitable, whether it has GPU or not.

#### Model Optimizations:
##### Quantization: 
<ul>
 <li> We’ll explore reducing model precision (16-bit or 8-bit) for faster inference with minimal accuracy loss (only for LLM).</li>
 <li> TensorRT: Use optimized operators for NVIDIA GPUs to accelerate inference.</li> 
</ul>
TensorRT and quantization are linked to the model serving process for efficient inference as Medical imaging models need to be accurate while being light enough to serve in real-time.

#### System Optimizations:
##### Load Balancing: 
Distribute inference requests across multiple servers to ensure scalability.
##### GPU Utilization: 
Use Ray for job distribution across GPUs to maximize resource use.

Ray and load balancing ensure that we can scale concurrently without compromising latency. With multiple doctors accessing the system simultaneously, it is essential to distribute load and optimize GPU utilization.


#### Evaluation and Monitoring
##### Offline Evaluation: 
Immediately after model training, we perform offline evaluations focusing on the recall metric to prioritize minimizing false negatives, which is critical for healthcare applications. We evaluate models on pneumonia, and tuberculosis detection using test images, respectively. </br>
All evaluation metrics, including recall, is  be logged using the MLFlow metric logger for tracking.

Model meeting the required recall (0.85) threshold will be automatically registered in the MLFlow model registry for deployment. If performance is inadequate, the model will be flagged for retraining.



##### Load Testing: 
We will use FastAPI to simulate multiple concurrent inference requests. We will deploy the model to the staging environment and use FastAPI’s async capabilities to handle 50-100 concurrent requests. The system’s performance will be evaluated by measuring latency, throughput, and resource usage (CPU/GPU/memory) during both batch and online inference tasks.

There is a fastapi endpoint with passphrase when probed will benchmark the service.


##### Online Monitoring AND Data Drift:
We are logging and visualizing metrics of triton server, fastapi server like inference latency, prediction latency (difference is end to end and just trition inference). 

We are also logging resource utilization and model degradation, data drift, confidence average, cummulative confidence prediction class distribution, etc.



##### Canary testing & Continuous feedback: 
In canary testing, the model will be deployed to a small subset of real users, allowing us to evaluate its performance in a live environment. The continuous feedback loop will collect doctor feedback on predictions, using this data to retrain the model periodically and improve its accuracy over time.

`Difficulty Points` Attempting from Unit-6 & Unit-7:
<ul> <li> We will build a dashboard for the engineers to be able to look into the data drift during re-training to ensure that the model is working as expected.</li> 
 <li>We will also monitor the model through the same dashboard to check the model health. </li>
</ul>

#### Data Pipeline 

##### Persistent Storage in Docker:
Data (X-rays, CT scans, and model artifacts) will be stored in Docker volumes for scalable and isolated storage.
PostgreSQL for patient, doctor, and interaction data will run in a Docker container, with data stored in persistent volumes to ensure durability and ease of management. This database will manage:
<ul>
 <li><strong>Patient Information</strong>: ID, name, age, medical history.</li>
 <li><strong>Doctor Information</strong>: ID, name, specialties, and consultation records.</li>
 <li><strong>Interactions</strong>: Records of patient visits and model predictions.</li>
 <li><strong>Training artifacts</strong>, models, and container images will be similarly managed in Docker to ensure portability and flexibility.</li>
</ul>


##### Data Pipelines
Transform data by applying feature engineering for structured data and validation for both. <br>
###### Clean and preprocess both structured and unstructured data:
<ul>
 <li><strong>Data Cleaning</strong>: Remove missing values from structured data and preprocess images (resize, normalize, augment).</li>
 <li><strong>Feature Engineering</strong>: Derive features like age from patient data, ensure image-label mappings are correct.</li>
 <li><strong>Data Validation</strong>: Ensure structured data conforms to schemas, and image formats are valid.</li>
</ul>

Load data into PostgreSQL for structured data <br>
To maintain data integrity and ensure reproducibility, we will implement data versioning using MLFlow for tracking datasets used in training and re-training.<br>

###### Online Data Pipeline: 
For real-time inference, we will set up a streaming pipeline using Kafka to handle live data (incoming X-ray images) for processing, cleaning, and inference. We will also simulate real-time data for testing and training purposes.

`Difficulty Points` Attempting from Unit-6 & Unit-7:
<ul> <li> We plan to implement an interactive and comprehensive data dashboard </li></ul>

#### Continuous X pipeline<br>
##### Infrastructure-as-code: 
Used Terraform to define, provision, and manage the entire cloud infrastructure.
Used to provision resources in two resources:
1) KVM_TACC -
   Instructions to provision resources:
   - cd terraform/kvm_tacc
   - Update clouds.yaml
   - source chi_openrc.sh
   - terraform init
   - terraform plan
   - terraform apply
2) CHI_UC -
   Instructions to provision resources:
   - cd terraform
   - Update clouds.yaml
   - source chi_openrc.sh
   - chmod +x reserve_gpu.sh
   - ./reserve_gpu.sh
   - cd chi_uc
   - terraform init (add reservation ID from output of reserve_gpu.sh)
   - terraform plan
   - terraform apply

Install ansible and kubespray dependencies <br>
Deploy Kubernetes using Ansible
- cd ansible
- ansible-playbook -i inventory.yml pre_k8s/pre_k8s_configure.yml
- ansible-playbook -i ../inventory/mycluster --become --become-user=root ./cluster.yml
- ansible-playbook -i inventory.yml post_k8s/post_k8s_configure.yml

##### Cloud-native:
Used services like minio, mlflow, postgres, grafana, prometheus
- Set up the ansible.cfg file with your IP
- ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml (This brings up all the services mentioned above)
- ansible-playbook -i inventory.yml argocd/workflow_build_init.yml
- ansible-playbook -i inventory.yml argocd/argocd_add_staging.yml
- ansible-playbook -i inventory.yml argocd/argocd_add_canary.yml
- ansible-playbook -i inventory.yml argocd/argocd_add_prod.yml
- ansible-playbook -i inventory.yml argocd/workflow_templates_apply.yml


##### CI/CD and continuous training:
If model is degraded to 85% then re-training is triggered.

##### Staged deployment:
- If recall values are above 90 and validaton accuracy are above 90 then it is promote to staging
- If recall values are above 92 and validation accuracy is above 92 then it is promote to canary.
- If recall values are above 94 and validation accuracy is above 94 then it is promote to production.






 
