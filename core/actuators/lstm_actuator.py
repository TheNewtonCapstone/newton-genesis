import torch
import genesis as gs
from torch import nn


class LSTMActuatorModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMActuatorModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Pass through LSTM
        output = self.fc(lstm_out)  # Apply fully connected layer
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
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"],
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
