import serpent.cv
import serpent.utilities
import serpent.ocr

from serpent.game_agent import GameAgent
from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

from datetime import datetime

import numpy as np

import skimage.color
import skimage.measure

import os
import time
import gc
import collections


class SerpentTiamatXGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None
        self._reset_game_state()

    def setup_play(self):
        input_mapping = {
            "UP": [KeyboardKey.KEY_W],
            "LEFT": [KeyboardKey.KEY_A],
            "DOWN": [KeyboardKey.KEY_S],
            "RIGHT": [KeyboardKey.KEY_D],
            "UP-LEFT": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
            "UP-RIGHT": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
            "SHOOT": [KeyboardKey.KEY_SPACE]
        }

        self.key_mapping = {
            KeyboardKey.KEY_W.name: "UP",
            KeyboardKey.KEY_A.name: "LEFT",
            KeyboardKey.KEY_S.name: "DOWN",
            KeyboardKey.KEY_D.name: "RIGHT",
            KeyboardKey.KEY_SPACE.name: "SHOOT"
        }

        direction_action_space = KeyboardMouseActionSpace(
            direction_keys=[None, "UP", "LEFT", "DOWN", "RIGHT", "UP-LEFT", "UP-RIGHT"]
        )

        action_space = KeyboardMouseActionSpace(
            action_keys=[None, "SHOOT"]
        )

        direction_model_file_path = "datasets/tiamatx_direction_dqn_0_1_.h5".replace("/", os.sep)

        self.dqn_direction = DDQN(
            model_file_path=direction_model_file_path if os.path.isfile(direction_model_file_path) else None,
            input_shape=(100, 100, 4),
            input_mapping=input_mapping,
            action_space=direction_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            model_learning_rate=1e-4,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

        action_model_file_path = "datasets/tiamatx_action_dqn_0_1_.h5".replace("/", os.sep)

        self.dqn_action = DDQN(
            model_file_path=action_model_file_path if os.path.isfile(action_model_file_path) else None,
            input_shape=(100, 100, 4),
            input_mapping=input_mapping,
            action_space=action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            model_learning_rate=1e-4,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

    def handle_play(self, game_frame):
        gc.disable()

        if self.dqn_direction.first_run:
            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

            time.sleep(5)

            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

            self.dqn_direction.first_run = False
            self.dqn_action.first_run = False

            time.sleep(5)

            return None

        vessel_hp = self._measure_vessel_hp(game_frame)
        vessel_score = self._measure_vessel_score(game_frame)

        self.game_state["health"].appendleft(vessel_hp)
        self.game_state["score"].appendleft(vessel_score)

        if self.dqn_direction.frame_stack is None:
            full_game_frame = FrameGrabber.get_frames(
                [0],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE"
            ).frames[0]

            self.dqn_direction.build_frame_stack(full_game_frame.frame)
            self.dqn_action.frame_stack = self.dqn_direction.frame_stack
        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE"
            )

            if self.dqn_direction.mode == "TRAIN":
                reward_direction, reward_action = self._calculate_reward()

                self.game_state["run_reward_direction"] += reward_direction
                self.game_state["run_reward_action"] += reward_action

                self.dqn_direction.append_to_replay_memory(
                    game_frame_buffer,
                    reward_direction,
                    terminal=self.game_state["health"] == 0
                )

                self.dqn_action.append_to_replay_memory(
                    game_frame_buffer,
                    reward_action,
                    terminal=self.game_state["health"] == 0
                )

                # Every 2000 steps, save latest weights to disk
                if self.dqn_direction.current_step % 2000 == 0:
                    self.dqn_direction.save_model_weights(
                        file_path_prefix=f"datasets/tiamatx_direction"
                    )

                    self.dqn_action.save_model_weights(
                        file_path_prefix=f"datasets/tiamatx_action"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.dqn_direction.current_step % 20000 == 0:
                    self.dqn_direction.save_model_weights(
                        file_path_prefix=f"datasets/tiamatx_direction",
                        is_checkpoint=True
                    )

                    self.dqn_action.save_model_weights(
                        file_path_prefix=f"datasets/tiamatx_action",
                        is_checkpoint=True
                    )
            elif self.dqn_direction.mode == "RUN":
                self.dqn_direction.update_frame_stack(game_frame_buffer)
                self.dqn_action.update_frame_stack(game_frame_buffer)

            run_time = datetime.now() - self.started_at

            serpent.utilities.clear_terminal()

            print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
            print("")

            print("DIRECTION NEURAL NETWORK:\n")
            self.dqn_direction.output_step_data()

            print("")
            print("ACTION NEURAL NETWORK:\n")
            self.dqn_action.output_step_data()

            print("")
            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_direction'] + self.game_state['run_reward_action'], 2)}")
            print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(f"CURRENT HEALTH: {self.game_state['health'][0]}")
            print(f"CURRENT SCORE: {self.game_state['score'][0]}")
            print("")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")

            print("")
            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'})")
            print("")

            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")

            if self.game_state["health"][1] <= 0:
                serpent.utilities.clear_terminal()
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.dqn_direction.mode in ["TRAIN", "RUN"]:
                    # Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_direction.mode == "RUN"
                        }
                else:
                    self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
                    self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])

                self.game_state["current_run_steps"] = 0

                self.input_controller.handle_keys([])

                if self.dqn_direction.mode == "TRAIN":
                    for i in range(8):
                        serpent.utilities.clear_terminal()
                        print(f"TRAINING ON MINI-BATCHES: {i + 1}/8")
                        print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                        self.dqn_direction.train_on_mini_batch()
                        self.dqn_action.train_on_mini_batch()

                self.game_state["run_timestamp"] = datetime.utcnow()
                self.game_state["current_run"] += 1
                self.game_state["run_reward_direction"] = 0
                self.game_state["run_reward_action"] = 0
                self.game_state["run_predicted_actions"] = 0
                self.game_state["health"] = collections.deque(np.full((8,), 3), maxlen=8)
                self.game_state["score"] = collections.deque(np.full((8,), 0), maxlen=8)

                if self.dqn_direction.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        if self.dqn_direction.type == "DDQN":
                            self.dqn_direction.update_target_model()
                            self.dqn_action.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.dqn_direction.enter_run_mode()
                        self.dqn_action.enter_run_mode()
                    else:
                        self.dqn_direction.enter_train_mode()
                        self.dqn_action.enter_train_mode()

                self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                time.sleep(3)

                self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

                return None

        self.dqn_direction.pick_action()
        self.dqn_direction.generate_action()

        self.dqn_action.pick_action(action_type=self.dqn_direction.current_action_type)
        self.dqn_action.generate_action()

        keys = self.dqn_direction.get_input_values() + self.dqn_action.get_input_values()
        print("")

        print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name), keys))))

        self.input_controller.handle_keys(keys)

        if self.dqn_direction.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.dqn_direction.erode_epsilon(factor=2)
        self.dqn_action.erode_epsilon(factor=2)

        self.dqn_direction.next_step()
        self.dqn_action.next_step()

        self.game_state["current_run_steps"] += 1

    def _reset_game_state(self):
        self.game_state = {
            "health": collections.deque(np.full((8,), 3), maxlen=8),
            "score": collections.deque(np.full((8,), 0), maxlen=8),
            "run_reward_direction": 0,
            "run_reward_action": 0,
            "current_run": 1,
            "current_run_steps": 0,
            "current_run_health": 0,
            "current_run_score": 0,
            "run_predicted_actions": 0,
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "run_timestamp": datetime.utcnow(),
        }

    def _measure_vessel_hp(self, game_frame):
        hp_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["HP_AREA"])

        vessel_hp = 0
        max_ssim = 0

        print(f"VESSEL_HP BEFORE: {vessel_hp}")

        for name, sprite in self.game.sprites.items():
            for i in range(sprite.image_data.shape[3]):
                ssim = skimage.measure.compare_ssim(hp_area_frame, np.squeeze(sprite.image_data[..., :3, i]), multichannel=True)

                if ssim > max_ssim:
                    max_ssim = ssim
                    vessel_hp = int(name[-1])

        print(f"VESSEL_HP AFTER: {vessel_hp}")
        return vessel_hp

    def _measure_vessel_score(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["SCORE_AREA"])

        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")

        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5)

        count = 0

        for char in score:
            if char == '0':
                count = count + 1
            else:
                break

        score = score[count:]

        self.game_state["current_run_score"] = score

        return score

    def _calculate_reward(self):
        reward = 0

        reward += (-0.5 if self.game_state["health"][0] < self.game_state["health"][1] else 0.05)
        reward += (0.5 if (self.game_state["score"][0] - self.game_state["score"][1]) >= 100 else -0.05)

        return reward, reward
