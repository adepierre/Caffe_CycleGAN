SET PATH=%PATH%;D:/Libraries/Caffe/install/bin

"../bin/Release/Caffe_CycleGAN.exe" ^
--train=0 ^
--image_size=256 ^
--batch_size=1 ^
--generator_features=64 ^
--discriminator_features=64 ^
--n_resnet=9 ^
--folder_A="../Data/horse2zebra/testA" ^
--folder_B="../Data/horse2zebra/testB" ^
--weights_gen_AtoB=CycleGAN_H2Z.caffemodel ^
--weights_gen_BtoA=CycleGAN_Z2H.caffemodel ^
--output_folder_A=testA/ ^
--output_folder_B=testB/ ^