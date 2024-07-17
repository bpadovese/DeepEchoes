import torch

class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear",
                 device='cpu'):
        """
        Initializes the NoiseScheduler with the specified number of timesteps and beta schedule.

        Parameters:
            num_timesteps (int): Total number of diffusion timesteps.
            beta_start (float): Starting value for the beta scheduling.
            beta_end (float): Ending value for the beta scheduling.
            beta_schedule (str): Type of scheduling for beta values ('linear' or 'quadratic').
        """
        self.device = device
        self.num_timesteps = num_timesteps
        # Betas represent the variance of the noise added at each timestep.
        if beta_schedule == "linear":
            # Linear schedule: betas increase linearly from beta_start to beta_end
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            # Quadratic schedule: betas increase quadratically (squared)
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        # Alphas represent the portion of the original data retained at each timestep.
        self.alphas = 1.0 - self.betas

        # Cumulative product of alphas, used to track the overall retained portion of the original data at each timestep
        # The higher the timestep, the higher the cumulative noise
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # Previous cumulative product of alphas, shifted by one timestep
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        # Precomputed terms
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        """
        Reconstructs the original data x_0 from the noisy data x_t at a specific timestep t using the sampled noise. I find the name of the 
        function a little misleading. Our goal at the end is to get x0, but this method reconstructs an estimate of the data at a 
        previous timestep t-1 from its current noisy state at t. Each execution of this method in the sequence during the reverse diffusion 
        cycle brings the data progressively closer to its original x0, noise-free state as tt approaches zero.

        Parameters:
            x_t (torch.Tensor): The noisy data at timestep t.
            t (int): The current timestep index.
            noise (torch.Tensor): The noise tensor sampled at timestep t.

        Returns:
            torch.Tensor: An estimate of the original data x_0.
        """
        # Note that the function uses coefficients that are derived from the model’s understanding of how noise was added during the forward diffusion process.
        # This was statically set at the initialization of the nosie scheduler, that is, nit doesnt change
        s1 = self.sqrt_inv_alphas_cumprod[t].to(self.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].to(self.device)

        # Flexible reshaping based on the number of dimensions of x_t
        # If you have fixed dimentions, you don't really need this. you can directly reshape to the shape you wish. 
        # For instance, if your dataset is a 1D (batch_size, features) you can simply do .reshape(-1, 1)
        num_dimensions = x_t.dim()  # Get the number of dimensions in x_t
        reshape_dim = [-1] + [1] * (num_dimensions - 1)  # Create a reshape list
        
        s1 = s1.reshape(*reshape_dim) 
        s2 = s2.reshape(*reshape_dim)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        """
        Computes the posterior mean of the data at the previous timestep based on the noisy observation x_t at timestep t.
        This posterior mean is an estimate of x_{t-1}, not the completely denoised original data x_0.

        Parameters:
            x_0 (torch.Tensor): An estimate of the data from the previous timestep.
            x_t (torch.Tensor): The noisy data at timestep t.
            t (int): The current timestep index.

        Returns:
            torch.Tensor: The mean of the posterior distribution, representing an estimate of the data state at the previous timestep.
        """
        s1 = self.posterior_mean_coef1[t].to(self.device)
        s2 = self.posterior_mean_coef2[t].to(self.device)

        # Flexible reshaping based on the number of dimensions of x_0
        num_dimensions = x_0.dim()  # Get the number of dimensions in x_0
        reshape_dim = [-1] + [1] * (num_dimensions - 1)  # Create a reshape list
        
        s1 = s1.reshape(*reshape_dim) 
        s2 = s2.reshape(*reshape_dim)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        """
        Calculates the variance of the posterior distribution for x_t at timestep t.

        Parameters:
            t (int): The current timestep index.

        Returns:
            float: The variance of the posterior at timestep t.
        """
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        """
        Performs a step in the reverse diffusion process to predict the previous timestep's sample.

        Parameters:
            model_output (torch.Tensor): The model output (predicted noise).
            timestep (int): The current timestep index.
            sample (torch.Tensor): The current noisy sample.

        Returns:
            torch.Tensor: The predicted previous timestep sample.
        """
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        """
        Adds noise to the original data x_start according to the diffusion schedule at specified timesteps.

        Parameters:
            x_start (torch.Tensor): The original data.
            x_noise (torch.Tensor): The noise to be added.
            timesteps (torch.Tensor): The timesteps at which the noise should be added.

        Returns:
            torch.Tensor: The noisy data after adding the noise.
        """
        s1 = self.sqrt_alphas_cumprod[timesteps].to(self.device)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].to(self.device)
        
        # Flexible reshaping based on the number of dimensions of x_start
        num_dimensions = x_start.dim()  # Get the number of dimensions in x_start
        reshape_dim = [-1] + [1] * (num_dimensions - 1)  # Create a reshape list
        
        s1 = s1.reshape(*reshape_dim) 
        s2 = s2.reshape(*reshape_dim)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        """
        Returns the number of timesteps in the diffusion process.

        Returns:
            int: The total number of timesteps.
        """
        return self.num_timesteps