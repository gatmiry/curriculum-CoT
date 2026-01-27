
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader
from model import GPTConfig, GPT
import torch.nn.functional as F
import re

class test_composite_function():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    def fit(self):

        print('im here!!!!!!!!!')
        """Test Composition functions"""
        print("=" * 80)


        function_dim = self.cfg.function_dim
        flipping_subsets = self.cfg.flipping_subsets
        if [] in flipping_subsets:
            flipping_subsets.remove([])
        flipping_subsets = flipping_subsets + [[]]
        cot_length = function_dim * len(flipping_subsets)
        block_size = function_dim + cot_length  # function_dim input tokens + zero tokens + 1 output token
        n_layers = self.cfg.n_layers
        n_heads = self.cfg.n_heads
        n_embd = self.cfg.n_embd
        dropout = self.cfg.dropout
        batch_size = self.cfg.batch_size
        learning_rate = self.cfg.learning_rate
        max_steps = self.cfg.max_steps
        target_function_str = self.cfg.get("target_function_string", "1.0x0x1")
        

        config = GPTConfig(
            vocab_size=function_dim,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_embd=n_embd,
            dropout=dropout,
            cot_length=cot_length
        )
        torch.manual_seed(41)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = GPT(config)
        model = model.to(device)

        print('initialized model')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        dataset = CompositeFunctionDataset(num_samples=10000, num_vars=function_dim, cot_length=cot_length, device=device)
        print('initialized dataset')
        dataset.set_target_function_from_string(target_function_str)

        # breakpoint()

        # Set target function from config
        import wandb
        wandb.init(project="distributional-CoT", name="test_composite_function")
        wandb.config.update({
            "function_dim": function_dim,
            "flipping_subsets": flipping_subsets,
            "cot_length": cot_length,
            "block_size": block_size,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_embd": n_embd,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps
        })
        for step in range(max_steps):
            #print('step ', step)
            optimizer.zero_grad()
            batch, cot_outputs = dataset.get_batch(batch_size=batch_size, flipping_subsets=flipping_subsets)
            #print('step is ', step, ' cot_outputs shape is ', cot_outputs.shape)
            batch = batch.to(device)
            #print('batch type is', batch.dtype)
            cot_outputs = cot_outputs.to(device)
            output_vals, loss = model(batch, cot_outputs)
            loss.backward()
            optimizer.step()     
    
            # Save the model
            if step % 100 == 0:
                #rint( 'shape of output_vals', output_vals[:,-2].shape, 'shape of cot_outputs[:, -1]', cot_outputs[:,-1].shape)
                #assert output_vals[:,-2].shape == cot_outputs[:,-1].shape, f"Output values and cot outputs have different lengths {output_vals[:,-2].shape} != {cot_outputs[:,-1].shape}"
                
                print('batch ', batch[0, :], ' output_vals ', output_vals[0, :],  ' cot_outputs ', cot_outputs[0, :])
                final_loss = F.mse_loss(output_vals[:, -1], cot_outputs[:, -1])   
                print(f"Step {step} Loss: {loss.item()} Final Loss: {final_loss.item()}")
                wandb.log({
                    "loss": loss.item(),
                    "final_loss": final_loss.item()
                }, step=step)

            #test_input = torch.randint(0, 2, (1, function_dim)).to(device)
            #print(f"Test input: {test_input[0, :]}")
            #test_output = model.generate(test_input)
            #print(f"Test output: {test_output[0].item()}")


class CompositeFunctionDataset():
    """Dataset for composite function testing with variable transformations."""
    
    def __init__(self, num_samples: int, num_vars: int, cot_length: int, seed: int = 41, device: str = 'cuda'):
        """
        Initialize dataset with random binary samples.
        
        Args:
            num_samples: Number of samples to generate
            num_vars: Number of variables (dimensions) per sample
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.num_vars = num_vars
        # Generate random binary samples: -1 or +1
        self.zero_one_samples = torch.randint(0, 2, (num_samples, num_vars)).to(device)
        self.samples = (self.zero_one_samples * 2 - 1)
        self.target_function_terms = []  # List of lists of variable indices for each term
        self.target_function_coefficients = []  # List of coefficients for each term
        self.target_function = None  # Function that computes the boolean function value
        self.device = device

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a single sample."""
        return self.samples[idx]
    
    def set_target_function_from_string(self, function_str: str):
        """
        Parse a function string and extract variable indices and coefficients for each term.
        Creates and returns a function that computes the boolean function value.
        
        Args:
            function_str: String like "0.1x0x1 + 2.8x2x4x5" where terms are separated by "+"
                         Each term has a coefficient followed by variables in format "x<index>"
        
        Returns:
            A function that takes samples (tensor [batch_size, num_vars]) and returns 
            function values (tensor [batch_size]).
            The function computes: sum(coefficient * product(variables)) for each term.
            
        Example:
            func = dataset.set_target_function_from_string("0.1x0x1 + 2.8x2x4x5")
            samples = torch.tensor([[1, -1, 1, 1, 1, 1]])  # [1, 1] sample
            values = func(samples)  # Computes 0.1*(1*-1) + 2.8*(1*1*1) = -0.1 + 2.8 = 2.7
        """
        # Split by "+" to get individual terms
        terms = [term.strip() for term in function_str.split('+')]
        
        variable_lists = []
        coefficients = []
        
        for term in terms:
            # Extract coefficient (number before the first 'x', including decimals and negative signs)
            # Pattern matches: optional minus sign, digits, optional decimal point and more digits
            coeff_match = re.match(r'^([+-]?\d*\.?\d+)', term)
            
            if coeff_match:
                coefficient = float(coeff_match.group(1))
            else:
                # If no coefficient found, default to 1.0
                coefficient = 1.0
            
            # Extract all variable indices from the term (pattern: x followed by digits)
            # This will match "x0", "x1", "x2", etc.
            variables = re.findall(r'x(\d+)', term)
            
            # Convert string indices to integers
            var_indices = [int(var) for var in variables]
            
            if var_indices:  # Only add non-empty lists
                variable_lists.append(var_indices)
                coefficients.append(coefficient)
        
        # Store in instance variables for later use
        self.target_function_terms = variable_lists
        self.target_function_coefficients = coefficients
        
        # Define the boolean function that computes the function value given samples
        
        def compute_function(samples: torch.Tensor) -> torch.Tensor:

            batch_size = samples.shape[0]
            target_vals = torch.zeros(batch_size, dtype=samples.dtype, device=samples.device)
            for coeff, var_subset in zip(self.target_function_coefficients, self.target_function_terms):
                term_vals = torch.ones(batch_size, dtype=samples.dtype, device=samples.device)
                for var in var_subset:
                    term_vals = term_vals * samples[:, var]
                target_vals = target_vals + coeff * term_vals
            return target_vals
        # Store the function in instance variable
        self.target_function = compute_function
        
        return compute_function

    def get_batch(self, batch_size: int, flipping_subsets: list[list[int]] = None) -> torch.Tensor:
        """
        Get a batch of samples with optional transformations.
        
        Args:
            batch_size: Number of samples in the batch
            pinning_vars: List of variable indices to pin to 1
            flipping_vars: List of variable indices to flip
            
        Returns:
            Transformed batch tensor [batch_size, num_vars]
        """

        # Add the empty flipping subset
        if [] not in flipping_subsets:
            print('adding empty flipping subset')
            flipping_subsets.append([])
        # Randomly sample indices
        indices = torch.randint(0, self.num_samples, (batch_size,)).to(self.device)
        int_zero_one_batch = self.zero_one_samples[indices].clone()
        batch = self.samples[indices].clone().float()
        #print('samples type in get_batch is', self.samples.dtype)
        # Apply transformations
        cot_length = self.num_vars * len(flipping_subsets)
        cot_outputs = torch.ones(batch_size, cot_length, dtype=batch.dtype, device=batch.device)
        cot_tmp_batch = torch.ones(batch.shape[0], self.num_vars, dtype=batch.dtype, device=batch.device)
        for pinning_start_index in range(1, self.num_vars + 1):
            cot_tmp_batch[:, pinning_start_index - 1] = batch[:, pinning_start_index - 1]
            for i, flipping_subset in enumerate(flipping_subsets):
                cot_tmp_batch_flipping = cot_tmp_batch.clone()
                for var in flipping_subset:
                    if var < pinning_start_index:
                        cot_tmp_batch_flipping[:, var] = -cot_tmp_batch_flipping[:, var]
                #print('cot_tmp_batch_flipping ', cot_tmp_batch_flipping[0, :])
                result = self.target_function(cot_tmp_batch_flipping)
                #print('result ', result[0])
                cot_outputs[:, (pinning_start_index - 1) * len(flipping_subsets) + i] = result
        #batch_with_cot = torch.cat([batch, cot_outputs], dim=1)

        #h type before return is', batch.dtype)
        #print('batch ', batch[0, :], ' cot_outputs ', cot_outputs[0, :])
        return int_zero_one_batch, cot_outputs

        




