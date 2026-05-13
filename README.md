# Code for Deep Learning Interpretability Analysis

Code and data for 'Exploring Interpretability in Deep Learning Prediction of Successful Ablation Therapy for Atrial Fibrillation' study.

Please cite this papers:

https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1054401/full

Dependencies:

  > NumPy
  > captum
  > lime
  > cv2
  > skimage
  > Pytorch

Example to use FA map code:

<pre><code>
FA_type = 'GradCAM'
model_pth = 'C:/Users/test_user/Desktop/Code/Model.pt'
image_pth = 'C:/Users/test_user/Desktop/Code/real_1R71W.jpg'
strat_index = 0
FA_obj = FA(FA_type,model_pth,image_pth,strat_index)
test = FA_obj.run()

</code></pre>
