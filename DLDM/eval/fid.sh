# pth="1000_sample_epoch_209"
# for dim in 64 192 768 2048
# do
#     for sqc in T1 T2 FLAIR WB BB
#     do
#         echo " "
#         echo $pth" --- "$sqc" --- "$dim
#         python -m pytorch_fid /home/yhn/Meta_synthesis/data/Severance_FID/$sqc /home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/$pth/$sqc --dims $dim
#     done
# done

pth="140000_stylegan2_1000"

for sqc in T1 T2 FLAIR WB BB
do
    echo " "
    echo $pth" --- "$sqc
    python -m pytorch_fid /home/yhn/Meta_synthesis/data/Severance_FID/$sqc /home/yhn/Meta_synthesis/data/stylegan2/$sqc
done

