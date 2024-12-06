import numpy as np
from scipy.stats import entropy

class FriedRiceNN:
    def __init__(self):
        self.v_engine = "1.0.0"
        self.board = None

        # Initialize neural network parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize neural network parameters (weights and biases).
        These are stored in a dictionary for consistency with EngineLinear.
        """
        input_size = 25
        hidden_size = 16
        output_size = 1

        # Initialize parameters with small random values
        self.parameters = {
            'W1': np.random.randn(hidden_size, input_size) * 0.01,  # Weights: input to hidden
            'b1': np.zeros((hidden_size, 1)),                       # Biases: hidden layer
            'W2': np.random.randn(output_size, hidden_size) * 0.01,  # Weights: hidden to output
            'b2': np.zeros((output_size, 1))                        # Biases: output layer
        }

    def forward(self, board_state):
        """
        Perform a forward pass through the neural network.
        Args:
            board_state (np.ndarray): Flattened board state (25, 1).
        Returns:
            np.ndarray: Output of the neural network (1, 1).
        """
        # Input to hidden layer
        Z1 = np.dot(self.parameters['W1'], board_state) + self.parameters['b1']
        A1 = np.tanh(Z1)  # Activation function

        # Hidden to output layer
        Z2 = np.dot(self.parameters['W2'], A1) + self.parameters['b2']
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for output

        return A2

    def evaluate_board(self, board):
        """
        Evaluate the current board state using the neural network.
        Args:
            board (np.ndarray): 5x5 board state.
        Returns:
            float: Evaluation score for the board.
        """
        # Flatten the 5x5 board into a 25x1 vector
        board_state = board.flatten().reshape(-1, 1)

        # Use the forward method to compute the evaluation score
        score = self.forward(board_state)

        return score[0, 0]

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Apply mutations to the neural network parameters.
        Args:
            mutation_rate (float): Probability of mutation for each parameter.
            mutation_scale (float): Scale of mutation applied to parameters.
        """
        for key in self.parameters:
            mutation_mask = np.random.rand(*self.parameters[key].shape) < mutation_rate
            self.parameters[key] += mutation_mask * np.random.randn(*self.parameters[key].shape) * mutation_scale

    def decide_move(self, board_state):
        """
        Decide the next move based on the board state.
        Args:
            board_state (np.ndarray): Flattened board state (25, 1).
        Returns:
            int: Index of the best move.
        """
        score = self.forward(board_state)
        return score.argmax()

    def get_parameters(self):
        """
        Get the neural network parameters.
        Returns:
            dict: Dictionary of network parameters ('W1', 'b1', 'W2', 'b2').
        """
        return self.parameters

    def set_parameters(self, parameters):
        """
        Set the neural network parameters.
        Args:
            parameters (dict): Dictionary of parameters to set ('W1', 'b1', 'W2', 'b2').
        """
        for key in self.parameters:
            if key in parameters:
                self.parameters[key] = parameters[key]

    def load_params(self,file=""):
        return self.set_parameters(np.load(file))