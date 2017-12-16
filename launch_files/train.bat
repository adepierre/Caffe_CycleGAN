SET PATH=%PATH%;D:/Libraries/Caffe/install/bin

"../bin/Release/Caffe_CycleGAN.exe" ^
--train=1 ^
--image_size=256 ^
--batch_size=1 ^
--generator_features=64 ^
--discriminator_features=64 ^
--n_resnet=9 ^
--folder_A="../Data/horse2zebra/trainA" ^
--folder_B="../Data/horse2zebra/trainB" ^
--weights_gen_AtoB= ^
--weights_gen_BtoA= ^
--solver_generator=solver_generator.prototxt ^
--solver_discriminator=solver_discriminator.prototxt ^
--logfile=log.csv ^
--max_pool_size=50 ^
--lambda=10.0 ^
--start_epoch=0 ^
--end_epoch=100 ^
--snapshot_gen_AtoB= ^
--snapshot_gen_BtoA= ^
--snapshot_discr_A= ^
--snapshot_discr_B= ^
--weights_discr_A= ^
--weights_discr_B= 
