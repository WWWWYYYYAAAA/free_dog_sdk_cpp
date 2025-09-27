import time

# import mujoco.viewer
# import mujoco
import numpy as np
# from legged_gym import LEGGED_GYM_ROOT_DIR
import threading
import torch
import yaml
import mc_sdk_py
import time
import math



def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

# 定义全局变量 obs
obs = None

ctrl_f = [0,0] # ctrl_f key_f

def on_press(key):
    ctrl_f[1]=1
    if key == keyboard.Key.up:
        cmd[0] = 1.0
    elif key == keyboard.Key.down:
        cmd[0] = -1.0
    if key == keyboard.Key.left:
        if ctrl_f[0] == 0:
            cmd[1] = 1.0
        else:
            cmd[2] = 1.0
    elif key == keyboard.Key.right:
        if ctrl_f[0] == 0:
            cmd[1] = -1.0
        else:
            cmd[2] = -1.0
    if key == keyboard.Key.ctrl:
        ctrl_f[0] = 1

def on_release(key):
    ctrl_f[1]=0
    if key == keyboard.Key.up:
        cmd[0] = 0.0
    elif key == keyboard.Key.down:
        cmd[0] = 0.0
    if key == keyboard.Key.left:
        cmd[1] = 0.0
        cmd[2] = 0.0
    elif key == keyboard.Key.right:
        cmd[1] = 0.0
        cmd[2] = 0.0
    if key == keyboard.Key.ctrl:
        ctrl_f[0] = 0
        cmd[2] = 0.0

# def padctrl(cmdctrl):
#     # last_cmd = cmd
#     ratio = 0.1
#     values = gamepad.GetInput(joyL=1,joyR=1,trigL=1,trigR=1,buttons=1,hat=1,joyL_max=100,os='linux')
#     cmdctrl[0] = 1.0*values[0][1]/100*ratio+cmdctrl[0]*(1-ratio)
#     cmdctrl[1] = -1.0*values[0][0]/100*ratio+cmdctrl[1]*(1-ratio)
#     cmdctrl[2] = 1.0*values[1][1]/100*ratio+cmdctrl[2]*(1-ratio)
#     return cmdctrl
def padctrl():
    values = gamepad.GetInput(joyL=1,joyR=1,trigL=1,trigR=1,buttons=1,hat=1,joyL_max=100,os='linux')
    cmd[0] = float(1.0*float(values[0][1])/100.0)
    cmd[1] = float(-1.0*float(values[0][0])/100.0)
    cmd[2] = float(1.0*float(values[1][1])/100.0)


class MotorControl:
    def __init__(self):
        self.motor_func = mc_sdk_py.LowLevel()
        self.motor_func.initRobot("127.0.0.1", 43988, "127.0.0.1") #local_ip, local_port, dog_ip
        self.init_q_abad = [0.0] * 4
        self.init_q_hip = [0.0] * 4
        self.init_q_knee = [0.0] * 4
        self.duration = 2.0
        self.default_abad_pos = 0.0
        self.default_hip_pos = 2.0
        self.default_knee_pos = -2.7
        self.default_hip_pos_list = [0.6, 0.6, 1.15, 1.15]
        self.stage1_progress = 0.0
        self.stage2_progress = 0.0
        self.stage = 0
        self.stage2_start = False
        self.first_trigger = False

    def get_data_from_dog(self):
        ang_vel = np.array((self.motor_func.getBodyGyroX(), self.motor_func.getBodyGyroY(), self.motor_func.getBodyGyroZ()))
        # gravity_orientation = np.array((self.motor_func.getBodyAccX(), self.motor_func.getBodyAccY(), self.motor_func.getBodyAccZ()))*-1.0
        roll = self.motor_func.getRoll()
        pitch = self.motor_func.getPitch()
        # yaw = self.motor_func.getYaw()
        gravity_orientation = np.array([math.sin(pitch), -math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch)])
        
        motorstate = self.motor_func.getMotorState()
        q_abad = np.array(motorstate.q_abad)
        q_hip = np.array(motorstate.q_hip)
        q_knee = np.array(motorstate.q_knee)
        qj = np.column_stack((q_abad, q_hip, q_knee)).flatten()[[3,4,5,0,1,2,9,10,11,6,7,8]]
        dq_abad = np.array(motorstate.qd_abad)
        dq_hip = np.array(motorstate.qd_hip)
        dq_knee = np.array(motorstate.qd_knee)
        dqj = np.column_stack((dq_abad, dq_hip, dq_knee)).flatten()[[3,4,5,0,1,2,9,10,11,6,7,8]]
        
        return ang_vel, gravity_orientation, qj, dqj
    

    def send_action(self, action):
        action = action.reshape(4, 3)
        cmd_mc = mc_sdk_py.MotorCommand()
        # cmd_mc.q_des_abad[:] = action[[1, 0, 3, 2], 0]
        # cmd_mc.q_des_hip[:] = action[[1, 0, 3, 2], 1]
        # cmd_mc.q_des_knee[:] = action[[1, 0, 3, 2], 2]
        cmd_mc.q_des_abad[:] = np.clip(action[[1, 0, 3, 2], 0], -0.4, 0.4)
        cmd_mc.q_des_hip[:] = np.clip(action[[1, 0, 3, 2], 1], -1.0, 3.0)
        cmd_mc.q_des_knee[:] = np.clip(action[[1, 0, 3, 2], 2], -2.5, 0.5)
        # print(cmd_mc.q_des_knee[:])
        cmd_mc.kp_abad[:] = np.array([40, 40, 40, 40])
        cmd_mc.kp_hip[:] = np.array([40, 40, 40, 40])
        cmd_mc.kp_knee[:] = np.array([40, 40, 40, 40])
        cmd_mc.kd_abad[:] = np.ones(4) * 1.0
        cmd_mc.kd_hip[:] = np.ones(4) * 1.0
        cmd_mc.kd_knee[:] = np.ones(4) * 1.0
        
        ret = self.motor_func.sendMotorCmd(cmd_mc)
        if ret < 0:
            print("send cmd error")
            
    def EmergencyStop(self):
        cmd_mc = mc_sdk_py.MotorCommand()
        cmd_mc.q_des_abad[:] = np.zeros(4)
        cmd_mc.q_des_hip[:] = np.zeros(4)
        cmd_mc.q_des_knee[:] = np.zeros(4)
        # cmd_mc.q_des_knee[:] = np.clip(action[[1, 0, 3, 2], 2], -2.85, 0.6)
        # print(cmd_mc.q_des_knee[:])
        cmd_mc.kp_abad[:] = np.zeros(4)
        cmd_mc.kp_hip[:] = np.zeros(4)
        cmd_mc.kp_knee[:] = np.zeros(4)
        cmd_mc.kd_abad[:] = np.ones(4) 
        cmd_mc.kd_hip[:] = np.ones(4)
        cmd_mc.kd_knee[:] = np.ones(4)
        
        ret = self.motor_func.sendMotorCmd(cmd_mc)
        if ret < 0:
            print("send cmd error")

    
    def stand_smooth(self):
        self.stage1_progress = 0.0
        cnt = 0
        while True:
            # 获取机器狗数据
            state = self.motor_func.getMotorState()

            if self.motor_func.haveMotorData():
                cnt += 1
                if cnt == 3500:
                    break
                if not self.first_trigger:
                    self.first_trigger = True
                    for i in range(4):
                        self.init_q_abad[i] = state.q_abad[i]
                        self.init_q_hip[i] = state.q_hip[i]
                        self.init_q_knee[i] = state.q_knee[i]

                self.stage1_progress += 0.002
                ratio = self.stage1_progress / self.duration
                if ratio > 1.0:
                    ratio = 1.0
                    self.stage = 1
                
                if self.stage == 1:
                    self.default_abad_pos = 0.0
                    self.default_hip_pos = 0.85
                    self.default_knee_pos = -1.5
                    self.stage2_progress += 0.002
                    ratio = self.stage2_progress / self.duration
                    if ratio > 1.0:
                        ratio = 1.0
                    
                    if not self.stage2_start:
                        self.stage2_start = True
                        for i in range(4):
                            self.init_q_abad[i] = state.q_abad[i]
                            self.init_q_hip[i] = state.q_hip[i]
                            self.init_q_knee[i] = state.q_knee[i]

                cmd_mc = mc_sdk_py.MotorCommand()
                for i in range(4):
                    cmd_mc.q_des_abad[i] = ratio * self.default_abad_pos + (1.0 - ratio) * self.init_q_abad[i]
                    if self.stage == 1:
                        cmd_mc.q_des_hip[i] = ratio * self.default_hip_pos_list[i] + (1.0 - ratio) * self.init_q_hip[i]
                    else:
                        cmd_mc.q_des_hip[i] = ratio * self.default_hip_pos + (1.0 - ratio) * self.init_q_hip[i]
                    cmd_mc.q_des_knee[i] = ratio * self.default_knee_pos + (1.0 - ratio) * self.init_q_knee[i]
                    cmd_mc.kp_abad[i] = 40
                    cmd_mc.kp_hip[i] = 40
                    cmd_mc.kp_knee[i] = 40
                    cmd_mc.kd_abad[i] = 1
                    cmd_mc.kd_hip[i] = 1
                    cmd_mc.kd_knee[i] = 1

                ret = self.motor_func.sendMotorCmd(cmd_mc)
                if ret < 0:
                    print("send cmd error")

            time.sleep(0.002)  # 等待 2 毫秒
    def run(self):
        while True:
            # 获取机器狗数据
            state = self.motor_func.getMotorState()

            if self.motor_func.haveMotorData():
                if not self.first_trigger:
                    self.first_trigger = True
                    for i in range(4):
                        self.init_q_abad[i] = state.q_abad[i]
                        self.init_q_hip[i] = state.q_hip[i]
                        self.init_q_knee[i] = state.q_knee[i]

                self.stage1_progress += 0.002
                ratio = self.stage1_progress / self.duration
                if ratio > 1.0:
                    ratio = 1.0
                    self.stage = 1
                
                if self.stage == 1:
                    self.default_abad_pos = 0.0
                    self.default_hip_pos = 0.8
                    self.default_knee_pos = -1.5
                    self.stage2_progress += 0.002
                    ratio = self.stage2_progress / self.duration
                    if ratio > 1.0:
                        ratio = 1.0
                    
                    if not self.stage2_start:
                        self.stage2_start = True
                        for i in range(4):
                            self.init_q_abad[i] = state.q_abad[i]
                            self.init_q_hip[i] = state.q_hip[i]
                            self.init_q_knee[i] = state.q_knee[i]

                cmd_mc = mc_sdk_py.MotorCommand()
                for i in range(4):
                    cmd_mc.q_des_abad[i] = ratio * self.default_abad_pos + (1.0 - ratio) * self.init_q_abad[i]
                    cmd_mc.q_des_hip[i] = ratio * self.default_hip_pos + (1.0 - ratio) * self.init_q_hip[i]
                    cmd_mc.q_des_knee[i] = ratio * self.default_knee_pos + (1.0 - ratio) * self.init_q_knee[i]
                    cmd_mc.kp_abad[i] = 80
                    cmd_mc.kp_hip[i] = 80
                    cmd_mc.kp_knee[i] = 80
                    cmd_mc.kd_abad[i] = 1
                    cmd_mc.kd_hip[i] = 1
                    cmd_mc.kd_knee[i] = 1

                ret = self.motor_func.sendMotorCmd(cmd_mc)
                if ret < 0:
                    print("send cmd error")

            time.sleep(0.002)  # 等待 2 毫秒
    def stop(self):
        cmd_mc = mc_sdk_py.MotorCommand()
        for i in range(4):
            cmd_mc.q_des_abad[i] = 0.0
            cmd_mc.q_des_hip[i] = 2.0
            cmd_mc.q_des_knee[i] = -2.5
            cmd_mc.kp_abad[i] = 0.0
            cmd_mc.kp_hip[i] = 0.0
            cmd_mc.kp_knee[i] = 0.0
            cmd_mc.kd_abad[i] = 3.0
            cmd_mc.kd_hip[i] = 3.0
            cmd_mc.kd_knee[i] = 3.0
        flag = True
        send_msg_count = 0
        while flag:
            ret = self.motor_func.sendMotorCmd(cmd_mc)
            if ret < 0:
                print("send cmd error")
            send_msg_count += 1
            if send_msg_count > 1500:
                flag = False
            time.sleep(0.002)  # 等待 2 毫秒


target_dof_pos = np.zeros(12)
emergency_stop = 1
def send_task():
    global target_dof_pos
    global emergency_stop
    while emergency_stop and motor_control.motor_func.haveMotorData():
        motor_control.send_action(target_dof_pos)
        time.sleep(0.0011)
    return
        
if __name__ == "__main__":
    
    config_file = "zsl1_real.yaml"
    with open(f"./config/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]

        # simulation_duration = config["simulation_duration"]
        # simulation_dt = config["simulation_dt"]
        # control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_one_step_obs = config["num_one_step_obs"]
        
        # cmd = np.array(config["cmd_init"], dtype=np.float32)

    cmd = np.array([0.0,0.0,0.0])
    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    current_obs = np.zeros(num_one_step_obs, dtype=np.float32)

    #lowlevel mc
    print("Initializing...")
    motor_control = MotorControl()
    time.sleep(2)
    print("Initialization completed")

    policy = torch.jit.load(policy_path)
    policy.eval()
    #ctrl
    from pynput import keyboard

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    from F710GamePad import F710GamePad

    gamepad = F710GamePad()
    dt = 0.02
    try:
        emergency_stop = 1
        motor_control.stand_smooth()
        print("standed")
        control_cnt = 0
        sendData = threading.Thread(target=send_task)
        sendData.start()
        print("begin to send")
    
        while emergency_stop and motor_control.motor_func.haveMotorData():
            
            start = time.time()
            # if control_cnt % 20 == 0:
            control_cnt = 0
            ang_vel, gravity_orientation, qj, dqj = motor_control.get_data_from_dog()
            last_cmd = cmd
            if ctrl_f[1] == 0:
                # cmd = padctrl(cmd)
                padctrl()
            # print(current_obs[:3] )
            ratio = 0.99
            cmd[:2] = cmd[:2] + np.clip(last_cmd[:2]-cmd[:2], -0.01, 0.01) * (1 - ratio)
            
            # print(cmd)
            current_obs[:3] = cmd * cmd_scale
            current_obs[3:6] = ang_vel * ang_vel_scale
            # current_obs[3:6] = np.array([0,0,0])
            current_obs[6:9] = gravity_orientation
            # current_obs[6:9] = np.array([0,0,-1])
            current_obs[9 : 9 + num_actions] = (qj - default_angles) * dof_pos_scale
            current_obs[9 + num_actions : 9 + 2 * num_actions] = dqj * dof_vel_scale
            current_obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
            
            # 将当前观测数据添加到 obs 的开头，并将历史数据向前移动
            obs = np.concatenate((current_obs, obs[:-num_one_step_obs]))
            
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # policy inference
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles
                # print(current_obs)
            # motor_control.send_action(target_dof_pos)
            time.sleep(0.0095)
            end = time.time()
            dt = dt*0.9+ 0.1* (end-start)
            print("\r", 1/dt, cmd, end="")
            # control_cnt += 1
    
    except Exception as e:
        print(e)
        try:
            emergency_stop = 0
            sendData.stop()
        except:
            pass
        # for i in range(2000):
        #     motor_control.EmergencyStop()
        #     time.sleep(0.001)
        # time.sleep(3)
        motor_control.stop()
        time.sleep(1)