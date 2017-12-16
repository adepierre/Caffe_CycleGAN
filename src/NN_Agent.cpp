#include "NN_Agent.h"

#include <chrono>

NN_Agent::NN_Agent(const std::string &solver_generator_,
				   const std::string &solver_discriminator_,
				   const std::string &log_filename,
				   const int &max_pool_size_,
				   const float &lambda_cycle,
				   const std::string &snapshot_generator_AtoB,
				   const std::string &snapshot_generator_BtoA,
				   const std::string &snapshot_discriminator_A,
				   const std::string &snapshot_discriminator_B,
				   const std::string &weights_generator_AtoB,
				   const std::string &weights_generator_BtoA,
				   const std::string &weights_discriminator_A,
				   const std::string &weights_discriminator_B)
{
	mean_real_loss_A = 0.0f;
	mean_generated_loss_A = 0.0f;
	mean_real_loss_B = 0.0f;
	mean_generated_loss_B = 0.0f;
	mean_cyclic_loss_A = 0.0f;
	mean_cyclic_loss_B = 0.0f;

	lambda = lambda_cycle;

	max_pool_size = max_pool_size_;

	//Create caffe objects (solvers + nets)
	caffe::SolverParameter solver_param_generator, solver_param_discriminator;
	caffe::ReadProtoFromTextFileOrDie(solver_generator_, &solver_param_generator);
	caffe::ReadProtoFromTextFileOrDie(solver_discriminator_, &solver_param_discriminator);

	std::string snapshot_prefix = solver_param_generator.snapshot_prefix();
	solver_param_generator.set_snapshot_prefix(snapshot_prefix + "_AtoB");
	solver_generator_AtoB.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_generator));
	solver_param_generator.set_snapshot_prefix(snapshot_prefix + "_BtoA");
	solver_generator_BtoA.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_generator));
	snapshot_prefix = solver_param_discriminator.snapshot_prefix();
	solver_param_discriminator.set_snapshot_prefix(snapshot_prefix + "_A");
	solver_discriminator_A.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_discriminator));
	solver_param_discriminator.set_snapshot_prefix(snapshot_prefix + "_B");
	solver_discriminator_B.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param_discriminator));
	net_generator_AtoB = solver_generator_AtoB->net();
	net_generator_BtoA = solver_generator_BtoA->net();
	net_discriminator_A = solver_discriminator_A->net();
	net_discriminator_B = solver_discriminator_B->net();

	if (snapshot_generator_AtoB.empty())
	{
		std::cout << "Starting new training for generator A to B" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot_generator_AtoB << " for generator A to B net." << std::endl;
		solver_generator_AtoB->Restore(snapshot_generator_AtoB.c_str());
	}

	if (snapshot_generator_BtoA.empty())
	{
		std::cout << "Starting new training for generator B to A" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot_generator_BtoA << " for generator B to A net." << std::endl;
		solver_generator_BtoA->Restore(snapshot_generator_BtoA.c_str());
	}

	if (snapshot_discriminator_A.empty())
	{
		std::cout << "Starting new training for discriminator A" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot_discriminator_A << " for discriminator A net." << std::endl;
		solver_discriminator_A->Restore(snapshot_discriminator_A.c_str());
	}

	if (snapshot_discriminator_B.empty())
	{
		std::cout << "Starting new training for discriminator B" << std::endl;
	}
	else
	{
		std::cout << "Loading " << snapshot_discriminator_B << " for discriminator B net." << std::endl;
		solver_discriminator_B->Restore(snapshot_discriminator_B.c_str());
	}

	if (!weights_generator_AtoB.empty())
	{
		std::cout << "Copying generator A to B weights from ... " << weights_generator_AtoB << std::endl;
		net_generator_AtoB->CopyTrainedLayersFrom(weights_generator_AtoB);
	}

	if (!weights_generator_BtoA.empty())
	{
		std::cout << "Copying generator B to A weights from ... " << weights_generator_BtoA << std::endl;
		net_generator_BtoA->CopyTrainedLayersFrom(weights_generator_BtoA);
	}

	if (!weights_discriminator_A.empty())
	{
		std::cout << "Copying discriminator A weights from ... " << weights_discriminator_A << std::endl;
		net_discriminator_A->CopyTrainedLayersFrom(weights_discriminator_A);
	}

	if (!weights_discriminator_B.empty())
	{
		std::cout << "Copying discriminator B weights from ... " << weights_discriminator_B << std::endl;
		net_discriminator_B->CopyTrainedLayersFrom(weights_discriminator_B);
	}

	//Get input and output blobs
	input_generator_AtoB = net_generator_AtoB->blob_by_name("generator_input");
	output_generator_AtoB = net_generator_AtoB->blob_by_name("generator_output");
	cyclic_input_AtoB = net_generator_AtoB->blob_by_name("cyclic_input");
	cyclic_loss_AtoB = net_generator_AtoB->blob_by_name("cyclic_loss");
	input_generator_BtoA = net_generator_BtoA->blob_by_name("generator_input");
	output_generator_BtoA = net_generator_BtoA->blob_by_name("generator_output");
	cyclic_input_BtoA = net_generator_BtoA->blob_by_name("cyclic_input");
	cyclic_loss_BtoA = net_generator_BtoA->blob_by_name("cyclic_loss");

	//Set the cyclic loss weight
	cyclic_loss_AtoB->mutable_cpu_diff()[0] = lambda;
	cyclic_loss_BtoA->mutable_cpu_diff()[0] = lambda;

	input_discriminator_A = net_discriminator_A->blob_by_name("images_input");
	input_discriminator_B = net_discriminator_B->blob_by_name("images_input");
	labels_discriminator_A = net_discriminator_A->blob_by_name("labels_input");
	labels_discriminator_B = net_discriminator_B->blob_by_name("labels_input");
	loss_discriminator_A = net_discriminator_A->blob_by_name("loss");
	loss_discriminator_B = net_discriminator_B->blob_by_name("loss");

	if (solver_discriminator_A->iter() > 0 || solver_generator_AtoB->iter() > 0 || solver_discriminator_B->iter() > 0 || solver_generator_BtoA->iter() > 0)
	{
		log_file.open(log_filename, std::ofstream::out|std::ofstream::app);
	}
	else
	{
		log_file.open(log_filename, std::ofstream::out);
		log_file << "Iter;Real loss A;Real loss B;Fake loss A; Fake loss B;Cyclic loss A to B;Cyclic loss B to A" << std::endl;
	}

	rand_gen = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

NN_Agent::NN_Agent(const std::string &model_file,
				   const std::string &trained_file_AtoB,
				   const std::string &trained_file_BtoA)
{
	net_generator_AtoB.reset(new caffe::Net<float>(model_file, caffe::TEST));
	net_generator_BtoA.reset(new caffe::Net<float>(model_file, caffe::TEST));

	if (!trained_file_AtoB.empty())
	{
		net_generator_AtoB->CopyTrainedLayersFrom(trained_file_AtoB);
	}

	if (!trained_file_BtoA.empty())
	{
		net_generator_BtoA->CopyTrainedLayersFrom(trained_file_BtoA);
	}

	input_generator_AtoB = net_generator_AtoB->blob_by_name("generator_input");
	input_generator_BtoA = net_generator_BtoA->blob_by_name("generator_input");
	output_generator_AtoB = net_generator_AtoB->blob_by_name("generator_output");
	output_generator_BtoA = net_generator_BtoA->blob_by_name("generator_output");
}

NN_Agent::~NN_Agent()
{
}

void NN_Agent::Train(const std::vector<float> &A_input, const std::vector<float> &B_input)
{
	net_discriminator_A->ClearParamDiffs();
	net_discriminator_B->ClearParamDiffs();
	net_generator_AtoB->ClearParamDiffs();
	net_generator_BtoA->ClearParamDiffs();

	//TODO Merge create a generic subfunction to group A to B and B to A
	/*A TO B*/

	//Generate images with the generator
	caffe::caffe_copy(input_generator_AtoB->count(), A_input.data(), input_generator_AtoB->mutable_cpu_data());

	net_generator_AtoB->ForwardFromTo(0, net_generator_AtoB->layers().size() - 6);

	//Copy the images to the other generator
	caffe::caffe_copy(output_generator_AtoB->count(), output_generator_AtoB->cpu_data(), input_generator_BtoA->mutable_cpu_data());
	caffe::caffe_copy(input_generator_AtoB->count(), input_generator_AtoB->cpu_data(), cyclic_input_BtoA->mutable_cpu_data());

	//Compute the cyclic loss
	net_generator_BtoA->Forward();
	net_generator_BtoA->Backward();
	mean_cyclic_loss_A += cyclic_loss_BtoA->cpu_data()[0] * cyclic_loss_BtoA->cpu_diff()[0];

	//Copy the cyclic diffs in the first generator
	caffe::caffe_copy(input_generator_BtoA->count(), input_generator_BtoA->cpu_diff(), output_generator_AtoB->mutable_cpu_diff());

	//Compute the discriminator loss
	caffe::caffe_copy(output_generator_AtoB->count(), output_generator_AtoB->cpu_data(), input_discriminator_B->mutable_cpu_data());
	caffe::caffe_set(labels_discriminator_B->count(), 1.0f, labels_discriminator_B->mutable_cpu_data());

	//Set the loss weight to 2 for the discriminator (because Caffe's Euclidean loss is 1/2N)
	loss_discriminator_B->mutable_cpu_diff()[0] = 2.0f;
	net_discriminator_B->Forward();
	net_discriminator_B->Backward();
	caffe::caffe_axpy(output_generator_AtoB->count(), 1.0f, input_discriminator_B->cpu_diff(), output_generator_AtoB->mutable_cpu_diff());

	//Propagate the diffs inside the generator
	net_generator_AtoB->BackwardFromTo(net_generator_AtoB->layers().size() - 6, 0);

	//Compute the discriminator loss
	//Generated loss
	net_discriminator_B->ClearParamDiffs();

	//Reset the discriminator loss weight to 1 (because we want loss_D = (loss_fake + loss_real) / 2)
	loss_discriminator_B->mutable_cpu_diff()[0] = 1.0f;

	//Set the fake label
	caffe::caffe_set(labels_discriminator_B->count(), 0.0f, labels_discriminator_B->mutable_cpu_data());

	//If pool is used, train on a random image from it half of the time
	//If we don't use an image from the pool, we just have to forward on the last two layers
	if (max_pool_size > 0)
	{
		std::vector<float> current_fake_batch(output_generator_AtoB->cpu_data(), output_generator_AtoB->cpu_data() + output_generator_AtoB->count());
		
		//We train the discriminator with the current batch of fake data, which is already inside D input blob
		if (batch_pool_B.size() < max_pool_size)
		{
			batch_pool_B.push_back(current_fake_batch);
			net_discriminator_B->ForwardFromTo(net_discriminator_B->layers().size() - 2, net_discriminator_B->layers().size() - 1);
		}
		else
		{
			if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rand_gen) > 0.5f)
			{
				int random_index = std::uniform_int_distribution<int>(0, batch_pool_B.size() - 1)(rand_gen);
				caffe::caffe_copy(input_discriminator_B->count(), batch_pool_B[random_index].data(), input_discriminator_B->mutable_cpu_data());
				batch_pool_B[random_index] = current_fake_batch;
				net_discriminator_B->Forward();
			}
			else
			{
				net_discriminator_B->ForwardFromTo(net_discriminator_B->layers().size() - 2, net_discriminator_B->layers().size() - 1);
			}
		}
	}

	net_discriminator_B->Backward();
	mean_generated_loss_B += loss_discriminator_B->cpu_data()[0] * loss_discriminator_B->cpu_diff()[0];

	caffe::caffe_copy(input_discriminator_B->count(), B_input.data(), input_discriminator_B->mutable_cpu_data());
	caffe::caffe_set(labels_discriminator_B->count(), 1.0f, labels_discriminator_B->mutable_cpu_data());


	net_discriminator_B->Forward();
	net_discriminator_B->Backward();
	mean_real_loss_B += loss_discriminator_B->cpu_data()[0] * loss_discriminator_B->cpu_diff()[0];

	/*B TO A*/

	//Generate images with the generator
	caffe::caffe_copy(input_generator_BtoA->count(), B_input.data(), input_generator_BtoA->mutable_cpu_data());

	net_generator_BtoA->ForwardFromTo(0, net_generator_BtoA->layers().size() - 6);

	//Copy the images to the other generator
	caffe::caffe_copy(output_generator_BtoA->count(), output_generator_BtoA->cpu_data(), input_generator_AtoB->mutable_cpu_data());
	caffe::caffe_copy(input_generator_BtoA->count(), input_generator_BtoA->cpu_data(), cyclic_input_AtoB->mutable_cpu_data());

	//Compute the cyclic loss
	net_generator_AtoB->Forward();
	net_generator_AtoB->Backward();
	mean_cyclic_loss_B += cyclic_loss_AtoB->cpu_data()[0] * cyclic_loss_AtoB->cpu_diff()[0];

	//Copy the cyclic diffs in the first generator
	caffe::caffe_copy(input_generator_AtoB->count(), input_generator_AtoB->cpu_diff(), output_generator_BtoA->mutable_cpu_diff());

	//Compute the discriminator loss
	caffe::caffe_copy(output_generator_BtoA->count(), output_generator_BtoA->cpu_data(), input_discriminator_A->mutable_cpu_data());
	caffe::caffe_set(labels_discriminator_A->count(), 1.0f, labels_discriminator_A->mutable_cpu_data());

	//Set the loss weight to 2 for the discriminator (because Caffe's Euclidean loss is 1/2N)
	loss_discriminator_A->mutable_cpu_diff()[0] = 2.0f;
	net_discriminator_A->Forward();
	net_discriminator_A->Backward();
	caffe::caffe_axpy(output_generator_BtoA->count(), 1.0f, input_discriminator_A->cpu_diff(), output_generator_BtoA->mutable_cpu_diff());

	//Propagate the diffs inside the generator
	net_generator_BtoA->BackwardFromTo(net_generator_BtoA->layers().size() - 6, 0);

	//Compute the discriminator loss
	//Generated loss
	net_discriminator_A->ClearParamDiffs();

	//Reset the discriminator loss weight to 1 (because we want loss_D = (loss_fake + loss_real) / 2)
	loss_discriminator_A->mutable_cpu_diff()[0] = 1.0f;

	//Set the fake label
	caffe::caffe_set(labels_discriminator_A->count(), 0.0f, labels_discriminator_A->mutable_cpu_data());

	//If pool is used, train on a random image from it half of the time
	//If we don't use an image from the pool, we just have to forward on the last two layers
	if (max_pool_size > 0)
	{
		std::vector<float> current_fake_batch(output_generator_BtoA->cpu_data(), output_generator_BtoA->cpu_data() + output_generator_BtoA->count());

		//We train the discriminator with the current batch of fake data, which is already inside D input blob
		if (batch_pool_A.size() < max_pool_size)
		{
			batch_pool_A.push_back(current_fake_batch);
			net_discriminator_A->ForwardFromTo(net_discriminator_A->layers().size() - 2, net_discriminator_A->layers().size() - 1);
		}
		else
		{
			if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rand_gen) > 0.5f)
			{
				int random_index = std::uniform_int_distribution<int>(0, batch_pool_A.size() - 1)(rand_gen);
				caffe::caffe_copy(input_discriminator_A->count(), batch_pool_A[random_index].data(), input_discriminator_A->mutable_cpu_data());
				batch_pool_A[random_index] = current_fake_batch;
				net_discriminator_A->Forward();
			}
			else
			{
				net_discriminator_A->ForwardFromTo(net_discriminator_A->layers().size() - 2, net_discriminator_A->layers().size() - 1);
			}
		}
	}

	net_discriminator_A->Backward();
	mean_generated_loss_A += loss_discriminator_A->cpu_data()[0] * loss_discriminator_A->cpu_diff()[0];

	caffe::caffe_copy(input_discriminator_A->count(), A_input.data(), input_discriminator_A->mutable_cpu_data());
	caffe::caffe_set(labels_discriminator_A->count(), 1.0f, labels_discriminator_A->mutable_cpu_data());


	net_discriminator_A->Forward();
	net_discriminator_A->Backward();
	mean_real_loss_A += loss_discriminator_A->cpu_data()[0] * loss_discriminator_A->cpu_diff()[0];

	solver_generator_AtoB->ApplyUpdate();
	solver_generator_AtoB->iter_++;
	solver_generator_BtoA->ApplyUpdate();
	solver_generator_BtoA->iter_++;
	solver_discriminator_A->ApplyUpdate();
	solver_discriminator_A->iter_++;
	solver_discriminator_B->ApplyUpdate();
	solver_discriminator_B->iter_++;
	
	//Snapshot and display
	if ((solver_discriminator_A->iter() % solver_discriminator_A->param().snapshot() == 0))
	{
		solver_discriminator_A->Snapshot();
	}

	if ((solver_discriminator_B->iter() % solver_discriminator_B->param().snapshot() == 0))
	{
		solver_discriminator_B->Snapshot();
	}

	if ((solver_generator_AtoB->iter() % solver_generator_AtoB->param().snapshot() == 0))
	{
		solver_generator_AtoB->Snapshot();
	}

	if ((solver_generator_BtoA->iter() % solver_generator_BtoA->param().snapshot() == 0))
	{
		solver_generator_BtoA->Snapshot();
	}

	if ((solver_discriminator_A->iter() % solver_discriminator_A->param().display() == 0))
	{
		int number_of_iterations = solver_discriminator_A->param().display();
		std::cout << "Loss on real A data: " << mean_real_loss_A / number_of_iterations << " on generated A data: " << mean_generated_loss_A / number_of_iterations << std::endl;
		if (log_file.is_open())
		{
			log_file << solver_discriminator_A->iter() << ";" << mean_real_loss_A / number_of_iterations << ";;" << mean_generated_loss_A / number_of_iterations << ";;;" << std::endl;
		}
		mean_real_loss_A = 0.0f;
		mean_generated_loss_A = 0.0f;
	}

	if ((solver_discriminator_B->iter() % solver_discriminator_B->param().display() == 0))
	{
		int number_of_iterations = solver_discriminator_B->param().display();
		std::cout << "Loss on real B data: " << mean_real_loss_B / number_of_iterations << " on generated B data: " << mean_generated_loss_B / number_of_iterations << std::endl;
		if (log_file.is_open())
		{
			log_file << solver_discriminator_B->iter() << ";;" << mean_real_loss_B / number_of_iterations << ";;" << mean_generated_loss_B / number_of_iterations << ";;" << std::endl;
		}
		mean_real_loss_B = 0.0f;
		mean_generated_loss_B = 0.0f;
	}
	
	if ((solver_generator_AtoB->iter() % solver_generator_AtoB->param().display() == 0))
	{
		int number_of_iterations = solver_generator_AtoB->param().display();
		std::cout << "Cyclic loss on A data: " << mean_cyclic_loss_A / number_of_iterations << std::endl;
		if (log_file.is_open())
		{
			log_file << solver_generator_AtoB->iter() << ";;;;;" << mean_cyclic_loss_A / number_of_iterations << ";" << std::endl;
		}
		mean_cyclic_loss_A = 0.0f;
	}

	if ((solver_generator_BtoA->iter() % solver_generator_BtoA->param().display() == 0))
	{
		int number_of_iterations = solver_generator_BtoA->param().display();
		std::cout << "Cyclic loss on B data: " << mean_cyclic_loss_B / number_of_iterations << std::endl;
		if (log_file.is_open())
		{
			log_file << solver_generator_BtoA->iter() << ";;;;;;" << mean_cyclic_loss_B / number_of_iterations << std::endl;
		}
		mean_cyclic_loss_B = 0.0f;
	}
}

void NN_Agent::Snapshot()
{
	solver_discriminator_A->Snapshot();
	solver_discriminator_B->Snapshot();
	solver_generator_AtoB->Snapshot();
	solver_generator_BtoA->Snapshot();
}

std::vector<float> NN_Agent::GeneratorTransform(const std::vector<float> &generator_input, const bool is_AtoB)
{
	if (is_AtoB)
	{
		caffe::caffe_copy(input_generator_AtoB->count(), generator_input.data(), input_generator_AtoB->mutable_cpu_data());
		net_generator_AtoB->Forward();
		return std::vector<float>(output_generator_AtoB->cpu_data(), output_generator_AtoB->cpu_data() + output_generator_AtoB->count());
	}
	else
	{
		caffe::caffe_copy(input_generator_BtoA->count(), generator_input.data(), input_generator_BtoA->mutable_cpu_data());
		net_generator_BtoA->Forward();
		return std::vector<float>(output_generator_BtoA->cpu_data(), output_generator_BtoA->cpu_data() + output_generator_BtoA->count());
	}
}

int NN_Agent::Iter()
{
	return solver_discriminator_A->iter();
}