import torch
import genesis as gs
from torch import nn


class LSTMActuatorModel(nn.Module):
    def __init__(self, input_size=2, hidden_size1=32, hidden_size2=16, output_size=1, dropout=0.2):
        """
        LSTM-based neural network for quadruped torque approximation

        Args:
            input_size: Number of input features (position_error + velocity for each actuator)
            hidden_size1: Size of first LSTM layer
            hidden_size2: Size of second LSTM layer
            output_size: Number of outputs (torques for each actuator)
            dropout: Dropout probability
        """
        super(LSTMActuatorModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size1, 1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, 1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, output_size)

    def __str__(self):
        return f"LSTMActuator(input_size={self.lstm1.input_size}, hidden_size1={self.lstm1.hidden_size}, hidden_size2={self.lstm2.hidden_size}, output_size={self.out.out_features})"

    def forward(self, x):
        # First LSTM layer
        output, _ = self.lstm1(x)
        output = self.dropout1(output)

        # Second LSTM layer
        output, (hidden, _) = self.lstm2(output)
        output = self.dropout2(output)  # Take only the last time step output

        # Dense layers
        output = self.relu(self.fc1(output))
        output = self.out(output)

        return output


class LSTMActuator:
    def __init__(
            self,
            scene: gs.Scene,
            motor_model_path: str,
            model_params: dict,
            device: torch.device,
    ):
        self.scene = scene
        self.device = device
        self._model_path: str = motor_model_path
        self._model: LSTMActuatorModel = LSTMActuatorModel(
        ).to(self.device)

        self._vec_gear_ratios = 9
        self._vec_effort_limits = 3.6

    def build(
            self
    ) -> None:
        self._model.load_state_dict(
            torch.load(self._model_path, map_location=self.device)
        )
        self._model.eval()

        print(
            f"LSTMActuator constructed with model from {self._model_path} running on {self.device}."
        )

    def step(
            self,
            output_current_positions,
            output_target_positions,
            output_current_velocities,
    ):
        return self._update_efforts(
            output_current_positions,
            output_target_positions,
            output_current_velocities,
        )

    def _update_efforts(
            self,
            output_current_positions,
            output_target_positions,
            output_current_velocities,
    ):
        input_position_errors = (
                                        output_target_positions - output_current_positions
                                ) * self._vec_gear_ratios
        input_current_velocities = output_current_velocities * self._vec_gear_ratios

        input_tensor = torch.stack(
            (
                input_position_errors,
                input_current_velocities,
            ),
            dim=1,
        )

        with torch.no_grad():
            computed_input_efforts = self._model(input_tensor).squeeze(1)

        # but the computed efforts are from the input's perspective, so we multiply again to get the output's efforts
        self._computed_output_efforts = computed_input_efforts * self._vec_gear_ratios

        applied_input_efforts = self._process_efforts(self._computed_output_efforts)
        return applied_input_efforts

    def _process_efforts(self, input_efforts):
        return torch.clamp(
            input_efforts,
            min=-self._vec_effort_limits,
            max=self._vec_effort_limits,
        )
