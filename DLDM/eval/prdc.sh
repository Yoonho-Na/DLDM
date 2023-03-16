# pth="1000_sample_epoch_209"

# for sqc in T1 T2 FLAIR WB BB
# do
#     echo " "
#     echo $pth" --- "$sqc
#     prdc_cli -r '/home/yhn/Meta_synthesis/data/Severance_FID/'$sqc -f '/home/yhn/Meta_synthesis/Project/models/disentangled-latent-diffusion_expand_dim/'$pth/$sqc -t 'R' -d 'cuda' -k 5
# done

pth="140000_stylegan2_1000"

for sqc in T1 T2 FLAIR WB BB
do
    echo " "
    echo $pth" --- "$sqc
    prdc_cli -r '/home/yhn/Meta_synthesis/data/Severance_FID/'$sqc -f '/home/yhn/Meta_synthesis/data/stylegan2/'$sqc -t 'R' -d 'cuda' -k 5

done

