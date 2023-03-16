import gdown, os, tarfile

pretrained_folder = 'pretrained_DLDM'
os.makedirs(pretrained_folder, exist_ok=True)
google_path = 'https://drive.google.com/uc?id='
file_id = '1JriniwO28vw_eRtN08gvHuc_EM6PlLDc'
output_name = pretrained_folder+'/DLDM_pretrained.tar.gz'
# gdown.download(google_path+file_id,output_name,quiet=False)
file = tarfile.open(output_name)
file.extractall(pretrained_folder)
file.close()
print('Pretrained weights extract done')