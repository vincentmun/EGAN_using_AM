# EGAN_using_AM

# Set-Up 
## 1.environment requirements:
To run this code, you need:  
- PyTorch 1.12.0+
- TensorFlow 2.2.0+
- cuda 11.3+  

Other requirements are in requirements.txt 

<!-- install code  -->
<pre><code>pip install -r requirements.txt
</code></pre>

## 2.datasets
We used CIFAR-10, CIFAR-100, STL-10 and ImageNet32 datasets. The default path for both datasets is `./datasets/{datasets name}`. Therefore, you should move or download required datasets in this path.


# Retrain the discovered architecture reported in the paper
## Fully train GAN on CIFAR-10
```bash
bash ./scripts/train_arch_cifar10.sh
```

or run the following commands

```bash
python fully_train_arch.py \
--gpu_ids 0 \
--num_workers 16 \
--gen_bs 256 \
--dis_bs 128 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_epoch_G 120 \
--n_critic 5 \
--arch arch_cifar10 \
--draw_arch False \
--genotypes_exp cifar10_D.npy \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 50000 \
--exp_name arch_train_cifar10 \
--data_path ./datasets/cifar10 \
--genotype_of_G EAMGAN_G.npy \
--use_basemodel_D False
```

- If you want to train on CIFAR-100 or ImageNet32, change the `--dataset` and `--data_path` parameters to match the respective dataset.

## Fully train GAN on STL-10
```bash
bash ./scripts/train_arch_stl10.sh
```

or run the following commands

```bash
python fully_train_arch.py \
--gpu_ids 0 \
--num_workers 16 \
--gen_bs 256 \
--dis_bs 128 \
--dataset stl10 \
--bottom_width 6 \
--img_size 48 \
--max_epoch_G 120 \
--n_critic 5 \
--arch arch_cifar10 \
--draw_arch False \
--genotypes_exp stl10_D.npy \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0003 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 50000 \
--exp_name arch_train_stl10 \
--data_path ./datasets/stl10 \
--genotype_of_G EAMGAN_G.npy \
--use_basemodel_D False
```
