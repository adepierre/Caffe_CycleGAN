#pragma once

#include <string>
#include <vector>
#include <random>

#include <caffe/caffe.hpp>


class NN_Agent
{
public:
	/**
	* \brief Create an agent for training, initialize nets and solvers
	* \param solver_generator_ Caffe solver for generator networks (*.prototxt file)
	* \param solver_discirminator_ Caffe solver for discriminator networks (*.prototxt file)
	* \param log_filename File in which the logs will be written
	* \param max_pool_size_ Maximum size of the pool (deactivated if 0)
	* \param lambda_cycle Weight of the cyclic loss
	* \param snapshot_generator_AtoB Caffe snapshot file to resume training of the generator A to B (*.solverstate file)
	* \param snapshot_generator_BtoA Caffe snapshot file to resume training of the generator B to A (*.solverstate file)
	* \param snapshot_discriminator_A Caffe snapshot file to resume training of the discriminator A (*.solverstate file)
	* \param snapshot_discriminator_B Caffe snapshot file to resume training of the discriminator B (*.solverstate file)
	* \param weights_generator_AtoB Caffe weights file to fine tune generator A to B (*.caffemodel)
	* \param weights_generator_BtoA Caffe weights file to fine tune generator B to A (*.caffemodel)
	* \param weights_discriminator_A Caffe weights file to fine tune discriminator A (*.caffemodel)
	* \param weights_discriminator_B Caffe weights file to fine tune discriminator B (*.caffemodel)
	*/
	NN_Agent(const std::string &solver_generator_,
			 const std::string &solver_discriminator_,
			 const std::string &log_filename,
			 const int &max_pool_size_,
			 const float &lambda_cycle,
			 const std::string &snapshot_generator_AtoB = "",
			 const std::string &snapshot_generator_BtoA = "",
			 const std::string &snapshot_discriminator_A = "",
			 const std::string &snapshot_discriminator_B = "",
			 const std::string &weights_generator_AtoB = "",
			 const std::string &weights_generator_BtoA = "",
			 const std::string &weights_discriminator_A = "",
			 const std::string &weights_discriminator_B = "");

	/**
	* \brief Create an agent for testing, initialize the net
	* \param model_file Caffe model file for generator net (*.prototxt)
	* \param trained_file_AtoB Caffe caffemodel file to fill weights of the A to B generator (*.caffemodel)
	* \param trained_file_BtoA Caffe caffemodel file to fill weights of the B to A generator (*.caffemodel)
	*/
	NN_Agent(const std::string &model_file,
			 const std::string &trained_file_AtoB,
			 const std::string &trained_file_BtoA);

	~NN_Agent();

	/**
	* \brief Perform one train cycle on the whole net (generator+discriminator)
	* \param A_input One batch of input from A dataset
	* \param B_input One batch of input from B dataset
	*/
	void Train(const std::vector<float> &A_input, const std::vector<float> &B_input);

	/**
	* \brief Snapshots every networks
	*/
	void Snapshot();

	/**
	* \brief Transform images through the generator
	* \param generator_input One batch of input for the generator
	* \param is_AtoB True if we want to use the generator which takes A as input, false for B
	* \return One batch of transformed data
	*/
	std::vector<float> GeneratorTransform(const std::vector<float> &generator_input, const bool is_AtoB);

	/**
	* \brief Get current solver iteration
	* \return Current solver iteration
	*/
	int Iter();

protected:
	//Common parameters
	boost::shared_ptr<caffe::Net<float> > net_generator_AtoB;
	boost::shared_ptr<caffe::Net<float> > net_generator_BtoA;

	boost::shared_ptr<caffe::Blob<float> > input_generator_AtoB;
	boost::shared_ptr<caffe::Blob<float> > input_generator_BtoA;
	boost::shared_ptr<caffe::Blob<float> > cyclic_input_AtoB;
	boost::shared_ptr<caffe::Blob<float> > cyclic_input_BtoA;
	boost::shared_ptr<caffe::Blob<float> > cyclic_loss_AtoB;
	boost::shared_ptr<caffe::Blob<float> > cyclic_loss_BtoA;
	boost::shared_ptr<caffe::Blob<float> > output_generator_AtoB;
	boost::shared_ptr<caffe::Blob<float> > output_generator_BtoA;
		
	//Parameters used for training
	boost::shared_ptr<caffe::Solver<float> > solver_discriminator_A;
	boost::shared_ptr<caffe::Solver<float> > solver_discriminator_B;
	boost::shared_ptr<caffe::Net<float> > net_discriminator_A;
	boost::shared_ptr<caffe::Net<float> > net_discriminator_B;
	boost::shared_ptr<caffe::Solver<float> > solver_generator_AtoB;
	boost::shared_ptr<caffe::Solver<float> > solver_generator_BtoA;

	boost::shared_ptr<caffe::Blob<float> > input_discriminator_A;
	boost::shared_ptr<caffe::Blob<float> > input_discriminator_B;
	boost::shared_ptr<caffe::Blob<float> > labels_discriminator_A;
	boost::shared_ptr<caffe::Blob<float> > labels_discriminator_B;
	boost::shared_ptr<caffe::Blob<float> > loss_discriminator_A;
	boost::shared_ptr<caffe::Blob<float> > loss_discriminator_B;

	std::ofstream log_file;

	std::vector<std::vector<float> > batch_pool_A;
	std::vector<std::vector<float> > batch_pool_B;
	int max_pool_size;

	float lambda;

	std::mt19937 rand_gen;
	
	float mean_real_loss_A;
	float mean_generated_loss_A;
	float mean_real_loss_B;
	float mean_generated_loss_B;
	float mean_cyclic_loss_A;
	float mean_cyclic_loss_B;
};


