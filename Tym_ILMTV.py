import numpy as np
from scipy import signal

class Tym_ILMTV:
    def __init__(self):
        self.v_engine = "0.0.1"
        self.board = None
        self.initialize_parameters()

    def initialize_parameters(self):
        # Inicializace parametrů konvoluční neuronové sítě
        # První konvoluční vrstva: 8 filtrů velikosti 3x3 (zvýšeno pro lepší detekci vzorů)
        self.conv1_filters = np.random.randn(8, 3, 3) * 0.1
        self.conv1_bias = np.zeros(8)

        # Druhá konvoluční vrstva: 16 filtrů (zvýšeno pro komplexnější vzory)
        self.conv2_filters = np.random.randn(16, 8, 3, 3) * 0.1
        self.conv2_bias = np.zeros(16)

        # Plně propojené vrstvy s větší kapacitou
        self.fc1_weights = np.random.randn(16 * 5 * 5, 32) * 0.1  # Zvětšeno na 32 neuronů
        self.fc1_bias = np.zeros(32)
        self.fc2_weights = np.random.randn(32, 1) * 0.1
        self.fc2_bias = np.zeros(1)

        # Uložení parametrů do dictionary pro pozdější použití
        self.parameters = {
            'conv1_filters': self.conv1_filters,
            'conv1_bias': self.conv1_bias,
            'conv2_filters': self.conv2_filters,
            'conv2_bias': self.conv2_bias,
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias
        }

    def relu(self, x):
        # Leaky ReLU - lepší gradient pro záporné hodnoty
        return np.where(x > 0, x, 0.01 * x)

    def tanh(self, x):
        # Upravený tanh pro agresivnější rozhodování
        return 1.5 * np.tanh(x)  # Zvětšení rozsahu pro extrémnější hodnoty

    def conv2d(self, input_data, filters, bias):
        # Implementace 2D konvoluce s více filtry
        n_filters = len(filters)
        if len(filters.shape) == 4:  # Pro druhou konvoluční vrstvu
            n_filters, in_channels, _, _ = filters.shape
            output = np.zeros((n_filters, input_data.shape[1], input_data.shape[2]))
            for i in range(n_filters):
                conv_sum = np.zeros_like(input_data[0])
                for j in range(in_channels):
                    conv_sum += signal.correlate2d(input_data[j], filters[i, j], mode='same')
                output[i] = conv_sum + bias[i]
        else:  # Pro první konvoluční vrstvu
            output = np.zeros((n_filters, input_data.shape[0], input_data.shape[1]))
            for i in range(n_filters):
                output[i] = signal.correlate2d(input_data, filters[i], mode='same') + bias[i]
        return output

    def forward(self, x):
        # Dopředný průchod sítí - od vstupu k výstupu
        # Přidáno škálování vstupu pro lepší využití aktivačních funkcí
        x = x * 2  # Škálování vstupu pro lepší využití rozsahu

        # První konvoluční vrstva + Leaky ReLU
        conv1_out = self.conv2d(x, self.conv1_filters, self.conv1_bias)
        conv1_activated = self.relu(conv1_out)

        # Druhá konvoluční vrstva + Leaky ReLU
        conv2_out = self.conv2d(conv1_activated, self.conv2_filters, self.conv2_bias)
        conv2_activated = self.relu(conv2_out)

        # Zploštění výstupu pro plně propojenou vrstvu
        flattened = conv2_activated.reshape(-1)

        # První plně propojená vrstva + Leaky ReLU
        fc1_out = np.dot(flattened, self.fc1_weights) + self.fc1_bias
        fc1_activated = self.relu(fc1_out)

        # Druhá plně propojená vrstva + upravený tanh
        fc2_out = np.dot(fc1_activated, self.fc2_weights) + self.fc2_bias
        output = self.tanh(fc2_out)

        return output

    def mutate(self, mutation_rate=0.1, mutation_scale=0.15):  # Zvýšená síla mutace
        # Agresivnější mutace pro rychlejší adaptaci
        for param_name in self.parameters:
            # Pravděpodobnost větších mutací je zvýšena
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            # Občasné velké mutace pro únik z lokálních minim
            large_mutation_mask = np.random.random(self.parameters[param_name].shape) < 0.05
            
            # Standardní mutace
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            # Velké mutace
            large_mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale * 3
            
            # Aplikace obou typů mutací
            self.parameters[param_name] += mutations * mutation_mask + large_mutations * large_mutation_mask

        self.conv1_filters = self.parameters['conv1_filters']
        self.conv1_bias = self.parameters['conv1_bias']
        self.conv2_filters = self.parameters['conv2_filters']
        self.conv2_bias = self.parameters['conv2_bias']
        self.fc1_weights = self.parameters['fc1_weights']
        self.fc1_bias = self.parameters['fc1_bias']
        self.fc2_weights = self.parameters['fc2_weights']
        self.fc2_bias = self.parameters['fc2_bias']

    def evaluate_board(self, board):
        # Ohodnocení stavu hrací desky pomocí CNN
        self.board = board.copy()
        
        # Přidána penalizace pro remízu (mírně negativní hodnota)
        output = self.forward(board)
        
        # Agresivnější škálování výstupu pro vyhraněnější rozhodnutí
        scaled_output = float(output * 12)  # Zvětšený rozsah pro extrémnější hodnocení
        
        # Pokud je hodnocení blízko nuly (potenciální remíza), mírně penalizujeme
        if abs(scaled_output) < 2:
            scaled_output -= 1  # Mírná penalizace pro remízové pozice
            
        return scaled_output

    def get_parameters(self):
        return self.parameters.copy()

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            parameters = {key: parameters[key] for key in parameters.files}
        
        self.parameters = parameters.copy()
        
        self.conv1_filters = self.parameters['conv1_filters']
        self.conv1_bias = self.parameters['conv1_bias']
        self.conv2_filters = self.parameters['conv2_filters']
        self.conv2_bias = self.parameters['conv2_bias']
        self.fc1_weights = self.parameters['fc1_weights']
        self.fc1_bias = self.parameters['fc1_bias']
        self.fc2_weights = self.parameters['fc2_weights']
        self.fc2_bias = self.parameters['fc2_bias']

    def load_params(self, file=""):
        return self.set_parameters(np.load(file))
