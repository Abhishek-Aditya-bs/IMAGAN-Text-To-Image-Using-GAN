# IMAGAN-Text-To-Image-Using-GAN 
Generation of images from text captions using Generative Adversarial Networks ( GAN’s ). Given a Text input the goal of the network is to generate a high-resolution image that matches text context.
 
 ## Scope
1. Synthesizing high-quality images from text descriptions is a challenging problem in computer vision and has many practical applications.
2. Samples generated by existing text-to-image approaches can roughly reflect the meaning of the given descriptions, but they fail to contain necessary details and vivid object parts.
3. Can be used to generate images that doesn’t exist in real life which can be useful in content creation.

 ## General Workflow of GAN Networks :
 ![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/GAN-Workflow.png)

## Work Carried Out 

1. Implemented StackGAN according to the [paper](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_StackGAN_Text_to_ICCV_2017_paper.html) and trained both the stages for 600 epochs on CUB dataset, but the outcome was not satisfying since 2 stage approach was harder to train and does not scale very well.

![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/StackGAN-Architecture.png)

2. Tried to modify the StackGAN architecture by modifying the generator in both the stages with the intention of generating better quality images, however due to the limitation of StackGAN the result obtained was not as expected.

3. Modified the Discriminator architecture of StackGAN by using a [U-NET](https://openaccess.thecvf.com/content_CVPR_2020/html/Schonfeld_A_U-Net_Based_Discriminator_for_Generative_Adversarial_Networks_CVPR_2020_paper.html) based approach. However, we could not find any improvements in our model.

![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/Stage-1-Output.png)

![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/Stage-2-Output.png)

4. Implemented Deep Fusion GAN [DF-GAN](https://arxiv.org/abs/2008.05865). This GAN architecture fixed the limitation of StackGAN and was able to generate high resolution images.

5. Modified the Generator architecture of DF-GAN by using Residual Dense Blocks [RDB](https://arxiv.org/pdf/1802.08797.pdf)

6. Due to the limitation of hardware resources, we used PyTorch Mixed Precision Training to effectively utilize the GPU and reduce the training time by 300% on V100 GPU.

![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/DF-GAN-Architecture.png)

# Images Generated by the new IMAGAN implemented using DF-GAN

![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/Op-1.png)


![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/Op-2.png)


![](https://github.com/Abhishek-Aditya-bs/IMAGAN-Text-To-Image-Using-GAN/blob/main/Images/Op-3.png)

# License
MIT


