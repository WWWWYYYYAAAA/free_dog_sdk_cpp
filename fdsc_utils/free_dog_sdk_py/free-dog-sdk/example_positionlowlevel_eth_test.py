from ucl.common import byte_print, decode_version, decode_sn, getVoltage, pretty_print_obj, lib_version
from ucl.lowState import lowState
from ucl.lowCmd import lowCmd
from ucl.unitreeConnection import unitreeConnection, LOW_WIFI_DEFAULTS, LOW_WIRED_DEFAULTS
from ucl.enums import GaitType, SpeedLevel, MotorModeLow
from ucl.complex import motorCmd, motorCmdArray
import time
import sys
import math
import numpy as np
from pprint import pprint
import threading
from F710GamePad import F710GamePad
import torch

   
# You can use one of the 3 Presets WIFI_DEFAULTS, LOW_CMD_DEFAULTS or HIGH_CMD_DEFAULTS.
# IF NONE OF THEM ARE WORKING YOU CAN DEFINE A CUSTOM ONE LIKE THIS:
#
# MY_CONNECTION_SETTINGS = (listenPort, addr_wifi, sendPort_high, local_ip_wifi)
# conn = unitreeConnection(MY_CONNECTION_SETTINGS)
d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5,
     'RR_0':6, 'RR_1':7, 'RR_2':8,
     'RL_0':9, 'RL_1':10, 'RL_2':11 }

PosStopF  = math.pow(10,9)
VelStopF  = 16000.0
LOWLEVEL  = 0xff
stand_pos = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5])
dt = 0.002
qInit = np.zeros(12, dtype=np.float32)
qDes = np.zeros(12, dtype=np.float32)
qj = np.zeros(12, dtype=np.float32)
dqj = np.zeros(12, dtype=np.float32)
sin_count = 0
rate_count = 0
Kp = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])
Kd = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print(f'Running lib version: {lib_version()}')
conn = unitreeConnection(LOW_WIRED_DEFAULTS)
conn.startRecv()
lcmd = lowCmd()
# lcmd.encrypt = True
lstate = lowState()
mCmdArr = motorCmdArray()
# Send empty command to tell the dog the receive port and initialize the connection
cmd_bytes = lcmd.buildCmd(debug=False)
conn.send(cmd_bytes)
data = conn.getData()

def get_data_from_dog():
    ang_vel = np.array(lstate.imu.gyroscope)
    # gravity_orientation = np.array((self.motor_func.getBodyAccX(), self.motor_func.getBodyAccY(), self.motor_func.getBodyAccZ()))*-1.0
    roll = lstate.imu.rpy[0]
    pitch = lstate.imu.rpy[1]
    # print("rpy", lstate.imu.rpy)
    # yaw = self.motor_func.getYaw()
    gravity_orientation = np.array([math.sin(pitch), -math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch)])
    qj = np.zeros(12, dtype=np.float32)
    dqj = np.zeros(12, dtype=np.float32)
    for _ in range(12):
        qj[_] = lstate.motorState[_].q
        dqj[_] = lstate.motorState[_].dq

    
    return ang_vel, gravity_orientation, qj[[3,4,5,0,1,2,9,10,11,6,7,8]], dqj[[3,4,5,0,1,2,9,10,11,6,7,8]]

for paket in data:
    print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')
    lstate.parseData(paket)
    print(f'SN [{byte_print(lstate.SN)}]:\t{decode_sn(lstate.SN)}')
    print(f'Ver [{byte_print(lstate.version)}]:\t{decode_version(lstate.version)}')
    print(f'SOC:\t\t\t{lstate.bms.SOC} %')
    print(f'Overall Voltage:\t{getVoltage(lstate.bms.cell_vol)} mv') #something is still wrong here ?!
    print(f'Current:\t\t{lstate.bms.current} mA')
    print(f'Cycles:\t\t\t{lstate.bms.cycle}')
    print(f'Temps BQ:\t\t{lstate.bms.BQ_NTC[0]} °C, {lstate.bms.BQ_NTC[1]}°C')
    print(f'Temps MCU:\t\t{lstate.bms.MCU_NTC[0]} °C, {lstate.bms.MCU_NTC[1]}°C')
    print(f'FootForce:\t\t{lstate.footForce}')
    print(f'FootForceEst:\t\t{lstate.footForceEst}')
    print(f'IMU Temp:\t\t{lstate.imu.temperature}')
    print(f'MotorState FR_0 MODE:\t\t{lstate.motorState[d["FR_0"]].mode}')
    print('+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=')

motiontime = 0
emergency_stop = 1

class SnedThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._running = True
    def run(self):
        global qDes
        global emergency_stop
        global Kp, Kd, mCmdArr, lcmd, conn
        while self._running and emergency_stop:
            mCmdArr.setMotorCmd('FR_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[0], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
            mCmdArr.setMotorCmd('FR_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[1], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
            mCmdArr.setMotorCmd('FR_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[2], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
            mCmdArr.setMotorCmd('FL_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[3], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
            mCmdArr.setMotorCmd('FL_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[4], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
            mCmdArr.setMotorCmd('FL_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[5], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
            mCmdArr.setMotorCmd('RR_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[6], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
            mCmdArr.setMotorCmd('RR_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[7], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
            mCmdArr.setMotorCmd('RR_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[8], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
            mCmdArr.setMotorCmd('RL_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[9], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
            mCmdArr.setMotorCmd('RL_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[10], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
            mCmdArr.setMotorCmd('RL_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[11], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
            lcmd.motorCmd = mCmdArr
            cmd_bytes = lcmd.buildCmd(debug=False)
            conn.send(cmd_bytes)
            time.sleep(0.002)
        return
    def stop(self):
        self._running = False
# thread = SnedThread()
# thread.start()
# time.sleep(5)  # 让线程运行5秒
# thread.stop()  # 停止线程
# thread.join()  # 等待线程结束
# print("线程已停止")

def send_task():
    global qDes
    global emergency_stop
    global Kp, Kd, mCmdArr, lcmd, conn
    while emergency_stop:
        mCmdArr.setMotorCmd('FR_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[0], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
        mCmdArr.setMotorCmd('FR_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[1], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
        mCmdArr.setMotorCmd('FR_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[2], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
        mCmdArr.setMotorCmd('FL_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[3], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
        mCmdArr.setMotorCmd('FL_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[4], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
        mCmdArr.setMotorCmd('FL_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[5], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
        mCmdArr.setMotorCmd('RR_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[6], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
        mCmdArr.setMotorCmd('RR_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[7], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
        mCmdArr.setMotorCmd('RR_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[8], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
        mCmdArr.setMotorCmd('RL_0',  motorCmd(mode=MotorModeLow.Servo, q=qDes[9], dq = 0, Kp = Kp[0], Kd = Kd[0], tau = -0.0))
        mCmdArr.setMotorCmd('RL_1',  motorCmd(mode=MotorModeLow.Servo, q=qDes[10], dq = 0, Kp = Kp[1], Kd = Kd[1], tau = 0.0))
        mCmdArr.setMotorCmd('RL_2',  motorCmd(mode=MotorModeLow.Servo, q=qDes[11], dq = 0, Kp = Kp[2], Kd = Kd[2], tau = 0.0))
        lcmd.motorCmd = mCmdArr
        cmd_bytes = lcmd.buildCmd(debug=False)
        conn.send(cmd_bytes)
        time.sleep(0.0017)
    return
def padctrl():
    global cmd
    values = gamepad.GetInput(joyL=1,joyR=1,trigL=1,trigR=1,buttons=1,hat=1,joyL_max=100,os='windows')
    cmd[0] = float(1.0*float(values[0][1])/100.0)
    cmd[1] = float(-1.0*float(values[0][0])/100.0)
    cmd[2] = float(-1.0*float(values[1][0])/100.0)
    
cmd = np.array([0.0,0.0,0.0])
gamepad = F710GamePad()
obs = np.zeros(270, dtype=np.float32)
current_obs = np.zeros(45, dtype=np.float32)
# lin_vel_scale: 2.0
# ang_vel_scale: 0.25 
# dof_pos_scale: 1.0 
# dof_vel_scale: 0.05 
# action_scale: 0.25
# cmd_scale: [2.0, 2.0, 0.75]
# num_actions: 12
# num_obs: 270
# num_one_step_obs: 45
cmd_scale = np.array([1.0, 1.0, 0.7])
lin_vel_scale=2.0
ang_vel_scale=0.25 
dof_pos_scale=1.0 
dof_vel_scale=0.05 
action_scale=0.25
num_actions=12
num_one_step_obs=45
default_angles = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
action = default_angles
policy_path = "./model/policy.pt"
policy = torch.jit.load(policy_path)
policy.eval()
try:
    while emergency_stop:
        start = time.time()
        data = conn.getData()
        for paket in data:
            lstate.parseData(paket)

        if( motiontime >= 0):

            # first, get record initial position
            if( motiontime >= 0 and motiontime < 5):
                if motiontime==1:
                    thread = SnedThread()
                    thread.start()
                    print("begin to send")
                for _ in range(12):
                    qDes[_] = lstate.motorState[_].q
                # qDes[0] = lstate.motorState[d['FR_0']].q
                # qDes[1] = lstate.motorState[d['FR_1']].q
                # qDes[2] = lstate.motorState[d['FR_2']].q
        
            # second, move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 50 and motiontime < 400):           # needs count to 200
                
                # Kp = [20, 20, 20]
                # Kd = [2, 2, 2]
                
                qDes = qDes + np.clip(stand_pos-qDes, -0.01, 0.01)
            
            if motiontime >= 400:
                ang_vel, gravity_orientation, qj, dqj = get_data_from_dog()
                # if motiontime%40==0:
                #     print("##########")
                #     print(f"{ang_vel}, {gravity_orientation}, {qj}, {dqj}")
                last_cmd = cmd
                    # cmd = padctrl(cmd)
                padctrl()
                # print(current_obs[:3] )
                cmd[:2] = last_cmd[:2] + np.clip(cmd[:2]-last_cmd[:2], -0.0005, 0.0005)
                
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
                qDes = target_dof_pos[[3,4,5,0,1,2,9,10,11,6,7,8]]
                    # print(current_obs)
                # motor_control.send_action(target_dof_pos)
        motiontime += 1
        time.sleep(0.017)
        end = time.time()
        dt = dt*0.9+ 0.1* (end-start)
        print("\r", 1/dt, cmd, end="")
except:
# except Exception as e:
    # print(e)
    emergency_stop=0
    try:
        thread.stop()  # 停止线程
        thread.join()  # 等待线程结束
    except:
        pass


        
