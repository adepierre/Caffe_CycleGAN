#include <gflags/gflags.h>

#include <NN_Agent.h>
#include <random>
#include <chrono>
#include <thread>
#include <sstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

//Common flags
DEFINE_int32(train, 1, "Whether we want to train the net or generate new data");
DEFINE_int32(image_size, 256, "Size of the images (16, 32, 64, 128)");
DEFINE_int32(batch_size, 1, "Number of samples in one pass");
DEFINE_int32(generator_features, 64, "Number of features for the generator convolution layers");
DEFINE_int32(discriminator_features, 64, "Number of features for the discriminator convolution layers");
DEFINE_int32(n_resnet, 9, "Number of resnet blocks for the generator");
DEFINE_string(folder_A, "../Data/horse2zebra/trainA", "Folder with A images");
DEFINE_string(folder_B, "../Data/horse2zebra/trainB", "Folder with B images");
DEFINE_string(weights_gen_AtoB, "", "Weights file to load (generator A to B)");
DEFINE_string(weights_gen_BtoA, "", "Weights file to load (generator B to A)");

//Training flags
DEFINE_string(solver_generator, "solver_generator.prototxt", "Solver file for generators");
DEFINE_string(solver_discriminator, "solver_discriminator.prototxt", "Solver file for discriminators");
DEFINE_string(logfile, "log.csv", "File to log the losses during training");
DEFINE_int32(max_pool_size, 50, "Maximum size of the memory pool. 0 to deactivate it");
DEFINE_double(lambda, 10.0, "Weight of the cyclic loss");
DEFINE_int32(start_epoch, 0, "Starting epoch number");
DEFINE_int32(end_epoch, 100, "Ending epoch number");
DEFINE_string(snapshot_gen_AtoB, "", "Snapshot file to resume training (generator A to B)");
DEFINE_string(snapshot_gen_BtoA, "", "Snapshot file to resume training (generator B to A)");
DEFINE_string(snapshot_discr_A, "", "Snapshot file to resume training (discriminator A)");
DEFINE_string(snapshot_discr_B, "", "Snapshot file to resume training (discriminator B)");
DEFINE_string(weights_discr_A, "", "Weights file to load (discriminator A)");
DEFINE_string(weights_discr_B, "", "Weights file to load (discriminator B)");

//Testing flags
DEFINE_string(output_folder_A, "../launch_files/testA/", "Output base folder for A transformed images");
DEFINE_string(output_folder_B, "../launch_files/testB/", "Output base folder for B transformed images");

/**
* \brief Load all the images from a folder into a vector
* \param folder Path of the folder containing the images
* \param destination Destination vector. The images are pushed at the end
* \param image_size Desired size of the images
*/
void LoadImagesFromFolder(const std::string &folder, std::vector<std::vector<float> > *destination, const int &image_size)
{
	for (boost::filesystem::directory_entry &file : boost::filesystem::directory_iterator(boost::filesystem::path(folder)))
	{
		if (boost::ends_with(file.path().string(), ".jpg") || boost::ends_with(file.path().string(), ".png") || boost::ends_with(file.path().string(), ".jpeg") || boost::ends_with(file.path().string(), ".bmp"))
		{
			cv::Mat img = cv::imread(file.path().string(), CV_LOAD_IMAGE_COLOR);
			cv::Mat resized_image;

			cv::resize(img, resized_image, cv::Size(image_size, image_size));

			cv::Mat img_float;
			resized_image.convertTo(img_float, CV_32F);
			img_float /= 127.5f;
			img_float -= cv::Scalar(1.0f, 1.0f, 1.0f);
			
			std::vector<cv::Mat> channels;
			cv::split(img_float, channels);

			std::vector<float> current_image;
			current_image.reserve(image_size * image_size * img_float.channels());

			for (int i = 0; i < img_float.channels(); ++i)
			{
				current_image.insert(current_image.end(), (float*)channels[i].datastart, (float*)channels[i].dataend);
			}
			destination->push_back(current_image);
		}
	}
}

/**
* \brief Convert network raw output into an image
* \param data Raw network output
* \param batch_size Number of images in the data
* \param image_size Size of the images
*/
cv::Mat BatchDataToImage(const std::vector<float> &data, const int &batch_size, const int &image_size)
{
	int number_image_per_line = std::ceil(std::sqrtf(batch_size));
	cv::Mat output = cv::Mat::zeros(cv::Size(number_image_per_line * image_size, number_image_per_line * image_size), CV_32FC3);

	for (int i = 0; i < batch_size; ++i)
	{
		std::vector<cv::Mat> channels;
		for (int c = 0; c < 3; ++c)
		{
			cv::Mat channel = cv::Mat(cv::Size(image_size, image_size), CV_32FC1);
			std::memcpy((float*)channel.data, data.data() + i * 3 * image_size * image_size + c * image_size * image_size, image_size * image_size * sizeof(float));
			channels.push_back(channel);
		}
		cv::Mat current_image;
		cv::merge(channels, current_image);

		current_image += cv::Scalar(1.0f, 1.0f, 1.0f);
		current_image *= 127.5f;

		current_image.copyTo(output(cv::Rect((i % number_image_per_line) * image_size, (i / number_image_per_line) * image_size, image_size, image_size)));
	}

	return output;
}

std::string ConvolutionLayer(const std::string &name, const std::string &bottom, const std::string &top, const int &n_features, const int &size, const int &stride, const int &pad, const bool train, const bool bias)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name << "\"" << "\n"
		<< "\t" << "type: \"Convolution\"" << "\n"
		<< "\t" << "bottom: \"" << bottom << "\"" << "\n"
		<< "\t" << "top: \"" << top << "\"" << "\n"
		<< "\t" << "convolution_param {" << "\n"
		<< "\t" << "\t" << "num_output: " << n_features << "\n"
		<< "\t" << "\t" << "kernel_size: " << size << "\n"
		<< "\t" << "\t" << "stride: " << stride << "\n"
		<< "\t" << "\t" << "pad: " << pad << "\n";
	if (bias)
	{
		layer_code
			<< "\t" << "\t" << "bias_term: true" << "\n";
	}
	else
	{
		layer_code
			<< "\t" << "\t" << "bias_term: false" << "\n";
	}

	if (train)
	{
		layer_code
			<< "\t" << "\t" << "weight_filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"gaussian\"" << "\n"
			<< "\t" << "\t" << "\t" << "std: 0.02" << "\n"
			<< "\t" << "\t" << "}" << "\n";
		if (bias)
		{
			layer_code
				<< "\t" << "\t" << "bias_filler {" << "\n"
				<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
				<< "\t" << "\t" << "\t" << "value: 0" << "\n"
				<< "\t" << "\t" << "}" << "\n";
		}
	}
	layer_code
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	return layer_code.str();
}

//Deconvolution implemented as NN-upsampling + convolution to avoid checkboard artifacts
std::string DeconvolutionLayer(const std::string &name, const std::string &bottom, const std::string &top, const int &input_features, const int &output_features, const int &size, const int &stride, const int &pad, const bool train, const bool bias)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name + "_upsample" << "\"" << "\n"
		<< "\t" << "type: \"Deconvolution\"" << "\n"
		<< "\t" << "bottom: \"" << bottom << "\"" << "\n"
		<< "\t" << "top: \"" << bottom + "_upsampled" << "\"" << "\n"
		<< "\t" << "convolution_param {" << "\n"
		<< "\t" << "\t" << "num_output: " << input_features << "\n"
		<< "\t" << "\t" << "group: " << input_features << "\n"
		<< "\t" << "\t" << "kernel_size: " << 2 << "\n"
		<< "\t" << "\t" << "stride: " << 2 << "\n"
		<< "\t" << "\t" << "pad: " << 0 << "\n" 
		<< "\t" << "\t" << "weight_filler {" << "\n"
		<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
		<< "\t" << "\t" << "\t" << "value: 1" << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "bias_term: false" << "\n"
		<< "\t" << "}" << "\n"
		<< "\t" << "param {" << "\n"
		<< "\t" << "\t" << "lr_mult: " << 0 << "\n"
		<< "\t" << "\t" << "decay_mult: " << 0 << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	layer_code
		<< ConvolutionLayer(name + "_conv", bottom + "_upsampled", top, output_features, size, stride, pad, train, bias);

	return layer_code.str();
}

std::string BatchNormLayer(const std::string &name, const std::string &bottom, const std::string &top)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name << "\"" << "\n"
		<< "\t" << "type: \"BatchNorm\"" << "\n"
		<< "\t" << "bottom: \"" << bottom << "\"" << "\n"
		<< "\t" << "top: \"" << top << "\"" << "\n"
		<< "\t" << "batch_norm_param {" << "\n"
		<< "\t" << "\t" << "use_global_stats: false" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	return layer_code.str();
}

std::string ScaleLayer(const std::string &name, const std::string &bottom, const std::string &top, const bool use_bias, const bool constant)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name << "\"" << "\n"
		<< "\t" << "type: \"Scale\"" << "\n"
		<< "\t" << "bottom: \"" << bottom << "\"" << "\n"
		<< "\t" << "top: \"" << top << "\"" << "\n"
		<< "\t" << "scale_param {" << "\n";
	if (!constant)
	{
		layer_code
			<< "\t" << "\t" << "filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"gaussian\"" << "\n"
			<< "\t" << "\t" << "\t" << "std: 0.02" << "\n"
			<< "\t" << "\t" << "\t" << "mean: 1.0" << "\n"
			<< "\t" << "\t" << "}" << "\n";
	}
	if (use_bias)
	{
		layer_code
			<< "\t" << "\t" << "bias_term: true" << "\n";

	}
	else
	{
		layer_code
			<< "\t" << "\t" << "bias_term: false" << "\n";
	}
	layer_code
		<< "\t" << "}" << "\n";
	if (constant)
	{
		layer_code
			<< "\t" << "param {" << "\n"
			<< "\t" << "\t" << "lr_mult: " << 0 << "\n"
			<< "\t" << "\t" << "decay_mult: " << 0 << "\n"
			<< "\t" << "}" << "\n";

		if (use_bias)
		{
			layer_code
				<< "\t" << "param {" << "\n"
				<< "\t" << "\t" << "lr_mult: " << 0 << "\n"
				<< "\t" << "\t" << "decay_mult: " << 0 << "\n"
				<< "\t" << "}" << "\n";
		}
	}

	layer_code
		<< "}" << "\n"
		<< "\n";

	return layer_code.str();
}

std::string ReLULayer(const std::string &name, const std::string &bottom, const std::string &top, const float &negative_slope = 0.0f)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name << "\"" << "\n"
		<< "\t" << "type: \"ReLU\"" << "\n"
		<< "\t" << "bottom: \"" << bottom << "\"" << "\n"
		<< "\t" << "top: \"" << top << "\"" << "\n";
	if (negative_slope != 0.0f)
	{
		layer_code
			<< "\t" << "relu_param {" << "\n"
			<< "\t" << "\t" << "negative_slope: " << negative_slope << "\n"
			<< "\t" << "}" << "\n";
	}
	layer_code
		<< "}" << "\n"
		<< "\n";

	return layer_code.str();
}

std::string SumLayer(const std::string &name, const std::vector<std::string> &bottoms, const std::string &top)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name << "\"" << "\n"
		<< "\t" << "type: \"Eltwise\"" << "\n";
	for (int i = 0; i < bottoms.size(); ++i)
	{
		layer_code
			<< "\t" << "bottom: \"" << bottoms[i] << "\"" << "\n";
	}
	layer_code
		<< "\t" << "top: \"" << top << "\"" << "\n"
		<< "\t" << "eltwise_param {" << "\n"
		<< "\t" << "\t" << "operation: SUM" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	return layer_code.str();
}

std::string ResnetBlock(const std::string &name, const std::string &bottom, const std::string &top, const int &input_features, const int &output_features, const bool train)
{
	std::stringstream block_code;

	block_code
		<< ConvolutionLayer(name + "_Conv1", bottom, bottom + "_Conv1", input_features, 3, 1, 1, train, false)
		<< BatchNormLayer(name + "_BN1", bottom + "_Conv1", bottom + "_Conv1")
		<< ScaleLayer(name + "_Scale1", bottom + "_Conv1", bottom + "_Conv1", true, false)
		<< ReLULayer(name + "_ReLU1", bottom + "_Conv1", bottom + "_Conv1", 0.2f)
		<< ConvolutionLayer(name + "_Conv2", bottom + "_Conv1", bottom + "_Conv2", output_features, 3, 1, 1, train, false)
		<< BatchNormLayer(name + "_BN2", bottom + "_Conv2", bottom + "_Conv2")
		<< ScaleLayer(name + "_Scale2", bottom + "_Conv2", bottom + "_Conv2", true, false);

	std::vector<std::string> sum_bottoms;
	sum_bottoms.push_back(bottom + "_Conv2");

	if (input_features != output_features)
	{
		block_code
			<< ConvolutionLayer(name + "_Conv3", bottom, bottom + "_Conv3", output_features, 1, 1, 0, train, false)
			<< BatchNormLayer(name + "_BN3", bottom + "_Conv3", bottom + "_Conv3")
			<< ScaleLayer(name + "_Scale3", bottom + "_Conv3", bottom + "_Conv3", true, false);
		sum_bottoms.push_back(bottom + "_Conv3");
	}
	else
	{
		sum_bottoms.push_back(bottom);
	}

	block_code
		<< SumLayer(name + "_Sum", sum_bottoms, top)
		<< ReLULayer(name + "_ReLU2", top, top, 0.2f);

	return block_code.str();
}

std::string ReshapeLayer(const std::string &name, const std::string &bottom, const std::string &top, const std::vector<int> &output_shape)
{
	std::stringstream layer_code;

	layer_code
		<< "layer {" << "\n"
		<< "\t" << "name: \"" << name << "\"" << "\n"
		<< "\t" << "type: \"Reshape\"" << "\n"
		<< "\t" << "bottom: \"" << bottom << "\"" << "\n"
		<< "\t" << "top: \"" << top << "\"" << "\n"
		<< "\t" << "reshape_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n";
	for (int i = 0; i < output_shape.size(); ++i)
	{
		layer_code
			<< "\t" << "\t" << "\t" << "dim: " << output_shape[i] << "\n";
	}
	layer_code
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	return layer_code.str();
}

/**
* \brief Create generator prototxt and save it in current directory as "generator.prototxt"
* \param train True if the net is for training, false otherwise
* \param batch_size Size of one batch in images
* \param image_size Size of the images
* \param n_features Number of features in the first convolution
* \param n_resnet Number of resnet block between the encoder and the decoder
*/
void CreateGeneratorPrototxt(const bool train, const int &batch_size, const int &image_size, const int &n_features, const int &n_resnet)
{
	std::ofstream prototxt("generator.prototxt");

	//Net header
	prototxt
		<< "name: \"Generator\"" << "\n"
		<< "force_backward: true" << "\n"
		//<< "debug_info: true" << "\n"
		<< "\n";

	//Input layer
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Input\"" << "\n"
		<< "\t" << "type: \"Input\"" << "\n"
		<< "\t" << "top: \"generator_input\"" << "\n"
		<< "\t" << "input_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << 3 << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Encoder
	prototxt
		<< ConvolutionLayer("Generator_Conv1", "generator_input", "generator_Conv1", n_features, 7, 1, 3, train, false)
		<< "#We don't use the computed mean and std at test time because" << "\n"
		<< "#it's calculated from both real and fake distribution which" << "\n" 
		<< "#is not what we want. As we use batch of size 1, it should" << "\n" 
		<< "#be equivalent to instance normalization." << "\n"
		<< BatchNormLayer("Generator_BN1", "generator_Conv1", "generator_Conv1")
		<< ScaleLayer("Generator_Scale1", "generator_Conv1", "generator_Conv1", true, false)
		<< ReLULayer("Generator_ReLU1", "generator_Conv1", "generator_Conv1", 0.2f)
		<< ConvolutionLayer("Generator_Conv2", "generator_Conv1", "generator_Conv2", 2 * n_features, 3, 2, 1, train, false)
		<< BatchNormLayer("Generator_BN2", "generator_Conv2", "generator_Conv2")
		<< ScaleLayer("Generator_Scale2", "generator_Conv2", "generator_Conv2", true, false)
		<< ReLULayer("Generator_ReLU2", "generator_Conv2", "generator_Conv2", 0.2f)
		<< ConvolutionLayer("Generator_Conv3", "generator_Conv2", "generator_Conv3", 4 * n_features, 3, 2, 1, train, false)
		<< BatchNormLayer("Generator_BN3", "generator_Conv3", "generator_Conv3")
		<< ScaleLayer("Generator_Scale3", "generator_Conv3", "generator_Conv3", true, false)
		<< ReLULayer("Generator_ReLU3", "generator_Conv3", "generator_Conv3", 0.2f);

	std::string current_bottom = "generator_Conv3";

	//Transformer
	for (int i = 0; i < n_resnet; ++i)
	{
		prototxt
			<< ResnetBlock("Generator_ResnetBlock_" + std::to_string(i), current_bottom, "output_resnet_" + std::to_string(i), 4 * n_features, 4 * n_features, train);
		current_bottom = "output_resnet_" + std::to_string(i);
	}

	//Decoder
	prototxt
		<< DeconvolutionLayer("Generator_Deconv1", current_bottom, "generator_Deconv1", 4 * n_features, 2 * n_features, 3, 1, 1, train, false)
		<< BatchNormLayer("Generator_BN4", "generator_Deconv1", "generator_Deconv1")
		<< ScaleLayer("Generator_Scale4", "generator_Deconv1", "generator_Deconv1", true, false)
		<< ReLULayer("Generator_ReLU4", "generator_Deconv1", "generator_Deconv1", 0.2f)
		<< DeconvolutionLayer("Generator_Deconv2", "generator_Deconv1", "generator_Deconv2", 2 * n_features, n_features, 3, 1, 1, train, false)
		<< BatchNormLayer("Generator_BN5", "generator_Deconv2", "generator_Deconv2")
		<< ScaleLayer("Generator_Scale5", "generator_Deconv2", "generator_Deconv2", true, false)
		<< ReLULayer("Generator_ReLU5", "generator_Deconv2", "generator_Deconv2", 0.2f)
		<< ConvolutionLayer("Generator_Conv4", "generator_Deconv2", "generator_pre_output", 1, 7, 1, 3, train, false);
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Generator_tanH_output\"" << "\n"
		<< "\t" << "type: \"TanH\"" << "\n"
		<< "\t" << "bottom: \"generator_pre_output\"" << "\n"
		<< "\t" << "top: \"generator_output\"" << "\n"
		<< "}" << "\n"
		<< "\n";

	//Cyclic loss (L1-Loss built up with caffe layers)
	if (train)
	{
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"Input_cyclic\"" << "\n"
			<< "\t" << "type: \"Input\"" << "\n"
			<< "\t" << "top: \"cyclic_input\"" << "\n"
			<< "\t" << "input_param {" << "\n"
			<< "\t" << "\t" << "shape {" << "\n"
			<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
			<< "\t" << "\t" << "\t" << "dim: " << 3 << "\n"
			<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
			<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		//L1-Loss
		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"Cyclic_loss_diff\"" << "\n"
			<< "\t" << "type: \"Eltwise\"" << "\n"
			<< "\t" << "bottom: \"generator_output\"" << "\n"
			<< "\t" << "bottom: \"cyclic_input\"" << "\n"
			<< "\t" << "top: \"cyclic_diff\"" << "\n"
			<< "\t" << "eltwise_param " << "{" << "\n"
			<< "\t" << "\t" << "operation: SUM" << "\n"
			<< "\t" << "\t" << "coeff: " << 1 << "\n"
			<< "\t" << "\t" << "coeff: " << -1 << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"Cyclic_loss_reduction_1\"" << "\n"
			<< "\t" << "type: \"Reduction\"" << "\n"
			<< "\t" << "bottom: \"cyclic_diff\"" << "\n"
			<< "\t" << "top: \"summed_diff\"" << "\n"
			<< "\t" << "reduction_param " << "{" << "\n"
			<< "\t" << "\t" << "operation: ASUM" << "\n"
			<< "\t" << "\t" << "axis: " << 1 << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"Cyclic_loss_scale\"" << "\n"
			<< "\t" << "type: \"Scale\"" << "\n"
			<< "\t" << "bottom: \"summed_diff\"" << "\n"
			<< "\t" << "top: \"scaled_diff\"" << "\n"
			<< "\t" << "scale_param " << "{" << "\n"
			<< "\t" << "\t" << "filler {" << "\n"
			<< "\t" << "\t" << "\t" << "type: \"constant\"" << "\n"
			<< "\t" << "\t" << "\t" << "value: " << 1.0f / (batch_size * 3 * image_size * image_size) << "\n"
			<< "\t" << "\t" << "}" << "\n"
			<< "\t" << "\t" << "axis: 0" << "\n"
			<< "\t" << "\t" << "bias_term: false" << "\n"
			<< "\t" << "}" << "\n"
			<< "\t" << "param " << "{" << "\n"
			<< "\t" << "\t" << "lr_mult: 0" << "\n"
			<< "\t" << "\t" << "decay_mult: 0" << "\n"
			<< "\t" << "}" << "\n"
			<< "}" << "\n"
			<< "\n";

		prototxt
			<< "layer {" << "\n"
			<< "\t" << "name: \"Cyclic_loss\"" << "\n"
			<< "\t" << "type: \"Reduction\"" << "\n"
			<< "\t" << "bottom: \"scaled_diff\"" << "\n"
			<< "\t" << "top: \"cyclic_loss\"" << "\n"
			<< "\t" << "reduction_param " << "{" << "\n"
			<< "\t" << "\t" << "operation: SUM" << "\n"
			<< "\t" << "}" << "\n"
			<< "\t" << "loss_weight: 1" << "\n"
			<< "}" << "\n"
			<< "\n";
	}

	prototxt.close();
}

/**
* \brief Create the discriminator prototxt and save it in current directory as "discriminator.prototxt"
* \param train True if the net is for training, false otherwise
* \param batch_size Size of one batch in images
* \param image_size Size of the images
* \param n_features Number of features in the first convolution
*/
void CreateDiscriminatorPrototxt(const bool train, const int &batch_size, const int &image_size, const int &n_features)
{
	std::ofstream prototxt("discriminator.prototxt");

	//Net header
	prototxt
		<< "name: \"Discriminator\"" << "\n"
		<< "force_backward: true" << "\n"
		//<< "debug_info: true" << "\n"
		<< "\n";

	//Input layer
	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Input\"" << "\n"
		<< "\t" << "type: \"Input\"" << "\n"
		<< "\t" << "top: \"images_input\"" << "\n"
		<< "\t" << "top: \"labels_input\"" << "\n"
		<< "\t" << "input_param {" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << 3 << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "\t" << "shape {" << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << batch_size << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << 1 << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size / 8 << "\n"
		<< "\t" << "\t" << "\t" << "dim: " << image_size / 8 << "\n"
		<< "\t" << "\t" << "}" << "\n"
		<< "\t" << "}" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt
		<< ConvolutionLayer("Discriminator_Conv1", "images_input", "Discriminator_Conv1", n_features, 4, 2, 1, train, false)
		<< "#We don't use the computed mean and std at test time because" << "\n"
		<< "#it's calculated from both real and fake distribution which" << "\n"
		<< "#is not what we want. As we use batch of size 1, it should" << "\n"
		<< "#be equivalent to instance normalization." << "\n"
		<< BatchNormLayer("Discriminator_BN1", "Discriminator_Conv1", "Discriminator_Conv1")
		<< ScaleLayer("Discriminator_Scale1", "Discriminator_Conv1", "Discriminator_Conv1", true, false)
		<< ReLULayer("Discriminator_ReLU1", "Discriminator_Conv1", "Discriminator_Conv1", 0.2f)
		<< ConvolutionLayer("Discriminator_Conv2", "Discriminator_Conv1", "Discriminator_Conv2", 2 * n_features, 4, 2, 1, train, false)
		<< BatchNormLayer("Discriminator_BN2", "Discriminator_Conv2", "Discriminator_Conv2")
		<< ScaleLayer("Discriminator_Scale2", "Discriminator_Conv2", "Discriminator_Conv2", true, false)
		<< ReLULayer("Discriminator_ReLU2", "Discriminator_Conv2", "Discriminator_Conv2", 0.2f)
		<< ConvolutionLayer("Discriminator_Conv3", "Discriminator_Conv2", "Discriminator_Conv3", 4 * n_features, 4, 2, 1, train, false)
		<< BatchNormLayer("Discriminator_BN3", "Discriminator_Conv3", "Discriminator_Conv3")
		<< ScaleLayer("Discriminator_Scale3", "Discriminator_Conv3", "Discriminator_Conv3", true, false)
		<< ReLULayer("Discriminator_ReLU3", "Discriminator_Conv3", "Discriminator_Conv3", 0.2f)
		<< ConvolutionLayer("Discriminator_Conv4", "Discriminator_Conv3", "Discriminator_Conv4", 8 * n_features, 3, 1, 1, train, false)
		<< BatchNormLayer("Discriminator_BN4", "Discriminator_Conv4", "Discriminator_Conv4")
		<< ScaleLayer("Discriminator_Scale4", "Discriminator_Conv4", "Discriminator_Conv4", true, false)
		<< ReLULayer("Discriminator_ReLU4", "Discriminator_Conv4", "Discriminator_Conv4", 0.2f);

	prototxt
		<< ConvolutionLayer("Discriminator_Conv5", "Discriminator_Conv4", "Discriminator_Output", 1, 1, 1, 0, train, false);


	prototxt
		<< ReshapeLayer("Reshape_input", "Discriminator_Output", "Discriminator_Reshaped_Output", std::vector<int>({ { -1 } }));

	prototxt
		<< ReshapeLayer("Reshape_labels", "labels_input", "labels_reshaped_input", std::vector<int>({ { -1 } }));

	prototxt
		<< "layer {" << "\n"
		<< "\t" << "name: \"Discriminator_loss\"" << "\n"
		<< "\t" << "type: \"EuclideanLoss\"" << "\n"
		<< "\t" << "bottom: \"Discriminator_Reshaped_Output\"" << "\n"
		<< "\t" << "bottom: \"labels_reshaped_input\"" << "\n"
		<< "\t" << "top: \"loss\"" << "\n"
		<< "\t" << "loss_weight: 1" << "\n"
		<< "}" << "\n"
		<< "\n";

	prototxt.close();
}

int main(int argc, char** argv)
{
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	//In test mode we don't need all the caffe stuff
	if (!FLAGS_train)
	{
		for (int i = 0; i < google::NUM_SEVERITIES; ++i)
		{
			google::SetLogDestination(i, "");
		}
	}
	else
	{
		google::LogToStderr();
	}

	std::mt19937 random_gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	CreateGeneratorPrototxt(FLAGS_train, FLAGS_batch_size, FLAGS_image_size, FLAGS_generator_features, FLAGS_n_resnet);

	//Read images and store data
	std::vector<std::vector<float> > images_A;
	std::vector<std::vector<float> > images_B;

	boost::filesystem::path folder_data_A = FLAGS_folder_A;
	boost::filesystem::path folder_data_B = FLAGS_folder_B;

	std::cout << "Reading data ..." << std::endl;

	if (boost::filesystem::is_empty(folder_data_A))
	{
		std::cerr << "Error, data folder A not found or empty." << std::endl;
		return -1;
	}

	LoadImagesFromFolder(folder_data_A.string(), &images_A, FLAGS_image_size);

	if (boost::filesystem::is_empty(folder_data_B))
	{
		std::cerr << "Error, data folder B not found or empty." << std::endl;
		return -1;
	}

	LoadImagesFromFolder(folder_data_B.string(), &images_B, FLAGS_image_size);

	//Train
	if (FLAGS_train)
	{
		CreateDiscriminatorPrototxt(FLAGS_train, FLAGS_batch_size, FLAGS_image_size, FLAGS_discriminator_features);

		NN_Agent nets(FLAGS_solver_generator, FLAGS_solver_discriminator, FLAGS_logfile, FLAGS_max_pool_size, 
					  FLAGS_lambda, FLAGS_snapshot_gen_AtoB, FLAGS_snapshot_gen_BtoA, FLAGS_snapshot_discr_A,
					  FLAGS_snapshot_discr_B, FLAGS_weights_gen_AtoB, FLAGS_weights_gen_BtoA, 
					  FLAGS_weights_discr_A, FLAGS_weights_discr_B);

		std::cout << "Networks ready." << std::endl;

		std::vector<std::vector<float> > generator_input_test_A;
		std::vector<std::vector<float> > generator_input_test_B;
		std::vector<cv::Mat> test_images_A;
		std::vector<cv::Mat> test_images_B;

		for (int i = 0; i < 4; ++i)
		{
			generator_input_test_A.push_back(images_A[i]);
			test_images_A.push_back(BatchDataToImage(images_A[i], 1, FLAGS_image_size));
			generator_input_test_B.push_back(images_B[i]);
			test_images_B.push_back(BatchDataToImage(images_B[i], 1, FLAGS_image_size));
		}

		int number_of_batch_in_epoch = std::min(images_A.size(), images_B.size()) / FLAGS_batch_size;

		std::cout << "One epoch is: " << number_of_batch_in_epoch << " iterations." << std::endl;

		//Number of epochs
		for (int epoch = FLAGS_start_epoch; epoch < FLAGS_end_epoch; ++epoch)
		{
			//Transform test data
			std::vector<cv::Mat> image_transformed_A;
			std::vector<cv::Mat> image_transformed_B;
			cv::Mat image_transformation_A = cv::Mat::zeros(cv::Size(4 * FLAGS_image_size, 2 * FLAGS_image_size), CV_32FC3);
			for (int i = 0; i < generator_input_test_A.size(); ++i)
			{
				std::vector<float> transformed_A = nets.GeneratorTransform(generator_input_test_A[i], true);
				BatchDataToImage(transformed_A, 1, FLAGS_image_size).copyTo(image_transformation_A(cv::Rect((2 * (i % 2) + 1)*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
				test_images_A[i].copyTo(image_transformation_A(cv::Rect((2 * (i % 2))*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
			}
			cv::Mat image_transformation_B = cv::Mat::zeros(cv::Size(4 * FLAGS_image_size, 2 * FLAGS_image_size), CV_32FC3);
			for (int i = 0; i < generator_input_test_B.size(); ++i)
			{
				std::vector<float> transformed_B = nets.GeneratorTransform(generator_input_test_B[i], false);
				BatchDataToImage(transformed_B, 1, FLAGS_image_size).copyTo(image_transformation_B(cv::Rect((2 * (i % 2) + 1)*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
				test_images_B[i].copyTo(image_transformation_B(cv::Rect((2 * (i % 2))*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
			}


			cv::imwrite("A_to_B_epoch_" + std::to_string(epoch) + ".png", image_transformation_A);
			cv::imwrite("B_to_A_epoch_" + std::to_string(epoch) + ".png", image_transformation_B);

			//Shuffle the indices of the images
			std::vector<int> indices_A;
			std::vector<int> indices_B;
			indices_A.reserve(images_A.size());
			indices_B.reserve(images_B.size());

			for (int i = 0; i < images_A.size(); ++i)
			{
				indices_A.push_back(i);
			}
			for (int i = 0; i < images_B.size(); ++i)
			{
				indices_B.push_back(i);
			}

			std::random_shuffle(indices_A.begin(), indices_A.end());
			std::random_shuffle(indices_B.begin(), indices_B.end());
						
			for (int batch = 0; batch < number_of_batch_in_epoch; ++batch)
			{
				std::vector<float> batch_data_A;
				std::vector<float> batch_data_B;

				for (int i = 0; i < FLAGS_batch_size; ++i)
				{
					batch_data_A.insert(batch_data_A.end(), images_A[indices_A[FLAGS_batch_size * batch + i]].begin(), images_A[indices_A[FLAGS_batch_size * batch + i]].end());
					batch_data_B.insert(batch_data_B.end(), images_B[indices_B[FLAGS_batch_size * batch + i]].begin(), images_B[indices_B[FLAGS_batch_size * batch + i]].end());
				}

				nets.Train(batch_data_A, batch_data_B);
			}
		}

		nets.Snapshot();

		std::vector<cv::Mat> image_transformed_A;
		std::vector<cv::Mat> image_transformed_B;
		cv::Mat image_transformation_A = cv::Mat::zeros(cv::Size(4 * FLAGS_image_size, 2 * FLAGS_image_size), CV_32FC3);
		for (int i = 0; i < generator_input_test_A.size(); ++i)
		{
			std::vector<float> transformed_A = nets.GeneratorTransform(generator_input_test_A[i], true);
			BatchDataToImage(transformed_A, 1, FLAGS_image_size).copyTo(image_transformation_A(cv::Rect((2 * (i % 2) + 1)*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
			test_images_A[i].copyTo(image_transformation_A(cv::Rect((2 * (i % 2))*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
		}
		cv::Mat image_transformation_B = cv::Mat::zeros(cv::Size(4 * FLAGS_image_size, 2 * FLAGS_image_size), CV_32FC3);
		for (int i = 0; i < generator_input_test_B.size(); ++i)
		{
			std::vector<float> transformed_B = nets.GeneratorTransform(generator_input_test_B[i], false);
			BatchDataToImage(transformed_B, 1, FLAGS_image_size).copyTo(image_transformation_B(cv::Rect((2 * (i % 2) + 1)*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
			test_images_B[i].copyTo(image_transformation_B(cv::Rect((2 * (i % 2))*FLAGS_image_size, i / 2 * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)));
		}


		cv::imwrite("A_to_B.png", image_transformation_A);
		cv::imwrite("B_to_A.png", image_transformation_B);

		return 0;
	}
	//Test
	else
	{
		NN_Agent nets("generator.prototxt", FLAGS_weights_gen_AtoB, FLAGS_weights_gen_BtoA);

		int index = 0;

		while (index < images_A.size())
		{
			std::vector<float> batch_A;
			for (int i = 0; i < std::min(FLAGS_batch_size, (int)(images_A.size() - index)); ++i)
			{
				batch_A.insert(batch_A.end(), images_A[index + i].begin(), images_A[index + i].end());
			}

			cv::Mat batch_input = BatchDataToImage(batch_A, FLAGS_batch_size, FLAGS_image_size);
			cv::Mat batch_output = BatchDataToImage(nets.GeneratorTransform(batch_A, true), FLAGS_batch_size, FLAGS_image_size);

			int number_image_per_line = std::ceil(std::sqrtf(FLAGS_batch_size));

			for (int i = 0; i < std::min(FLAGS_batch_size, (int)(images_A.size() - index)); ++i)
			{
				cv::Mat output = cv::Mat(FLAGS_image_size, 2 * FLAGS_image_size, CV_32FC3);
				batch_input(cv::Rect((i % number_image_per_line) * FLAGS_image_size, (i / number_image_per_line) * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)).copyTo(output(cv::Rect(0, 0, FLAGS_image_size, FLAGS_image_size)));
				batch_output(cv::Rect((i % number_image_per_line) * FLAGS_image_size, (i / number_image_per_line) * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)).copyTo(output(cv::Rect(FLAGS_image_size, 0, FLAGS_image_size, FLAGS_image_size)));
				cv::imwrite(FLAGS_output_folder_A + std::to_string(index + i) + ".png", output);
			}
			index += FLAGS_batch_size;
		}

		index = 0;
		while (index < images_B.size())
		{
			std::vector<float> batch_B;
			for (int i = 0; i < std::min(FLAGS_batch_size, (int)(images_B.size() - index)); ++i)
			{
				batch_B.insert(batch_B.end(), images_B[index + i].begin(), images_B[index + i].end());
			}

			cv::Mat batch_input = BatchDataToImage(batch_B, FLAGS_batch_size, FLAGS_image_size);
			cv::Mat batch_output = BatchDataToImage(nets.GeneratorTransform(batch_B, false), FLAGS_batch_size, FLAGS_image_size);

			int number_image_per_line = std::ceil(std::sqrtf(FLAGS_batch_size));

			for (int i = 0; i < std::min(FLAGS_batch_size, (int)(images_B.size() - index)); ++i)
			{
				cv::Mat output = cv::Mat(FLAGS_image_size, 2 * FLAGS_image_size, CV_32FC3);
				batch_input(cv::Rect((i % number_image_per_line) * FLAGS_image_size, (i / number_image_per_line) * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)).copyTo(output(cv::Rect(0, 0, FLAGS_image_size, FLAGS_image_size)));
				batch_output(cv::Rect((i % number_image_per_line) * FLAGS_image_size, (i / number_image_per_line) * FLAGS_image_size, FLAGS_image_size, FLAGS_image_size)).copyTo(output(cv::Rect(FLAGS_image_size, 0, FLAGS_image_size, FLAGS_image_size)));
				cv::imwrite(FLAGS_output_folder_B + std::to_string(index + i) + ".png", output);
			}
			index += FLAGS_batch_size;
		}
		
		return 0;
	}
}