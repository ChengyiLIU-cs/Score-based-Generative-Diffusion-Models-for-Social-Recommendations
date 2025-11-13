### *[TKDE'25] Score-based Generative Diffusion Models for Social Recommendations*

### **Abstract**
With the prevalence of social networks on online platforms, social recommendation has become a vital technique for enhancing personalized recommendations. The effectiveness of social recommendations largely relies on the social homophily assumption, which presumes that individuals with social connections often share similar preferences. However, this foundational premise has been recently challenged due to the inherent complexity and noise present in real-world social networks. In this paper, we tackle the low social homophily challenge from an innovative generative perspective, directly generating optimal user social representations that maximize consistency with collaborative signals. Specifically, we propose the Score-based Generative Model for Social Recommendation (SGSR), which effectively adapts the Stochastic Differential Equation (SDE)-based diffusion models for social recommendations. To better fit the recommendation context, SGSR employs a joint curriculum training strategy to mitigate challenges related to missing supervision signals and leverages self-supervised learning techniques to align knowledge across social and collaborative domains. Extensive experiments on real-world datasets demonstrate the effectiveness of our approach in filtering redundant social information and improving recommendation performance.

### **Quick Start**
Run the code with: python main.py --dataset ciao --core 0

### **Citation & References**
  volume={37},
  number={11},
  pages={6666-6679},
  keywords={Noise reduction;Collaboration;Training;Social networking (online);Diffusion models;Recommender systems;Noise;Diffusion processes;Adaptation models;Fans;Recommender systems;score-based generative diffusion models;social graph denoising},
  doi={10.1109/TKDE.2025.3600103}}
