# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import binarymaze_utils.maze_utils as maze_utils
import binarymaze_utils.traj_utils as traj_utils


class TrajectoryGenerator(object):
    def __init__(self, options, place_cells):
        self.options = options
        self.maze_size = 4
        self.place_cells = place_cells
        self.maze = maze_utils.NewMaze(self.maze_size)
        self.node_pos = np.zeros((len(self.maze.ru),2))
        for j,r in enumerate(self.maze.ru):
          self.node_pos[j,0] = self.maze.xc[r[-1]]
          self.node_pos[j,1]=self.maze.yc[r[-1]]
        #self.node_pos = (self.node_pos-np.array([3,3]))[:,::-1]
        self.wall_pos = self.maze.wa - np.mean(self.node_pos,axis=0)
        self.node_pos -= np.mean(self.node_pos,axis=0)
        height_ratio = (np.max(self.node_pos[:,0])-np.min(self.node_pos[:,0])+1)/self.options.box_height        
        self.node_pos[:,0] = self.node_pos[:,0]/height_ratio# - self.options.box_height/2
        width_ratio = (np.max(self.node_pos[:,1])-np.min(self.node_pos[:,1])+1)/self.options.box_width
        self.node_pos[:,1] = self.node_pos[:,1]/width_ratio# - self.options.box_width/2
        self.wall_pos[:,0] = self.wall_pos[:,0]/height_ratio
        self.wall_pos[:,1] = self.wall_pos[:,1]/width_ratio
        self.corridor_width = 1/width_ratio
        self.cell_len = self.node_pos[3,1] - self.node_pos[7,1] 

        self.leaf_node = np.arange(self.node_pos.shape[0]-np.power(2,self.maze_size)+1,self.node_pos.shape[0]+1,1)-1
        const = self.corridor_width/2
        self.node_shift = dict(
          {15:[-const,0],16:[const,0],17:[-const,0],18:[const,0],
            19:[-const,0],20:[const,0],21:[-const,0],22:[const,0],
            23:[-const,0],24:[const,0],25:[-const,0],26:[const,0],
            27:[-const,0],28:[const,0],29:[-const,0],30:[const,0]
        })
        self.shift_node_pos = self.node_pos.copy()
        for i_node in self.node_shift.keys():
          self.shift_node_pos[i_node] = self.shift_node_pos[i_node] + self.node_shift[i_node]

        #all wall position
        tmp_wall_pos = []
        for i_wall in range(len(self.wall_pos)-1):
          x = np.arange(np.min(self.wall_pos[i_wall:i_wall+2,0]),np.max(self.wall_pos[i_wall:i_wall+2,0]),self.options.box_height/100)[:,None]
          y = np.arange(np.min(self.wall_pos[i_wall:i_wall+2,1]),np.max(self.wall_pos[i_wall:i_wall+2,1]),self.options.box_height/100)[:,None]
          if len(x)==0:
            x = np.repeat(self.wall_pos[i_wall,0],len(y),axis=0)[:,None]
          if len(y)==0:
            y = np.repeat(self.wall_pos[i_wall,1],len(x),axis=0)[:,None]
          tmp_wall_pos.append(np.concatenate((x,y),axis=1))
        tmp_wall_pos = np.concatenate(tmp_wall_pos,axis=0)
        self.ori_wall_pos = np.unique(tmp_wall_pos,axis=0)



    def get_wall_pos(self,res):
          x_wall = ((self.ori_wall_pos[:,0]+self.options.box_width/2) /(self.options.box_width) * res)[:,None]
          y_wall = ((self.ori_wall_pos[:,1]+self.options.box_width/2) /(self.options.box_width) * res)[:,None]
          self.res_wall_pos = (np.concatenate((x_wall,y_wall),axis=1)).astype(int)
          return self.res_wall_pos

    def maze_randomwalk(self,steps: int,
                    random_seed: int,
                    mode='node') -> np.ndarray:
      """ Generate synthetic data
      Args:
        steps: number of steps for random walk
        mode: select cell or node
      Returns:
        List[nodes]
      """
  
      Traj = traj_utils.MakeRandomWalk(self.maze,
                            n = steps,
                            rs = random_seed)
      if mode == 'node':
        return Traj.no[0][:-1, 0]
      elif mode == 'cell':
        return Traj.ce
      return rw

    def pos2polarangle(self,pos):
      if pos[0]<0:
        angle = 180
      if pos[0]>0: 
        angle = 0
      if pos[1]<0:
        angle = 270
      if pos[1]>0:
        angle = 90
      return angle/180*np.pi
    

    def shrink_and_expand(self,x,t_len):
      n_batch = len(x)
      for i in range(n_batch):
        x[i] = x[i][:t_len]
      y = np.empty((n_batch,t_len,x[0].shape[1]))#batch x sequence x feature
      for i in range(n_batch):
        y[i,:,:] = x[i]
      
      return y

    def avoid_wall(self, position, hd, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:,0]
        y = position[:,1]
        dists = [box_width/2-x, box_height/2-y, box_width/2+x, box_height/2+y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4)*np.pi/2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2*np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi
        
        is_near_wall = (d_wall < self.border_region)*(np.abs(a_wall) < np.pi/2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall])*(np.pi/2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle


    def generate_trajectory(self, box_width, box_height, batch_size):
        '''Generate a random walk in a rectangular box'''
        if (self.options.traj_type == 'maze_env'):
          traj = self.maze_env_walk(box_width,box_height,batch_size)
        if (self.options.traj_type == 'maze_open_env'):
          traj = self.maze_env_open_walk(box_width,box_height,batch_size)
        if (self.options.traj_type == 'open_env'):
          traj = self.open_env_walk(box_width,box_height,batch_size)
        return traj

    def open_env_walk(self, box_width, box_height, batch_size):
        '''Generate a random walk in a rectangular box'''
        samples = self.options.sequence_length
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias 
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples+2, 2])
        head_dir = np.zeros([batch_size, samples+2])
        position[:,0,0] = np.random.uniform(-box_width/2, box_width/2, batch_size)
        position[:,0,1] = np.random.uniform(-box_height/2, box_height/2, batch_size)
        head_dir[:,0] = np.random.uniform(0, 2*np.pi, batch_size)
        velocity = np.zeros([batch_size, samples+2])
        
        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples+1])
        random_vel = np.random.rayleigh(b, [batch_size, samples+1])
        v = np.abs(np.random.normal(0, b*np.pi/2, batch_size))

        for t in range(samples+1):
            # Update velocity
            v = random_vel[:,t]
            turn_angle = np.zeros(batch_size)

            if not self.options.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:,t], head_dir[:,t], box_width, box_height)
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt*random_turn[:,t]

            # Take a step
            velocity[:,t] = v*dt
            update = velocity[:,t,None]*np.stack([np.cos(head_dir[:,t]), np.sin(head_dir[:,t])], axis=-1)
            position[:,t+1] = position[:,t] + update

            # Rotate head direction
            head_dir[:,t+1] = head_dir[:,t] + turn_angle

        # Periodic boundaries
        if self.options.periodic:
            position[:,:,0] = np.mod(position[:,:,0] + box_width/2, box_width) - box_width/2
            position[:,:,1] = np.mod(position[:,:,1] + box_height/2, box_height) - box_height/2

        head_dir = np.mod(head_dir + np.pi, 2*np.pi) - np.pi # Periodic variable

        traj = {}
        # Input variables
        traj['init_hd'] = head_dir[:,0,None]
        traj['init_x'] = position[:,1,0,None]
        traj['init_y'] = position[:,1,1,None]

        traj['ego_v'] = velocity[:,1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        traj['target_hd'] = head_dir[:,1:-1]
        traj['target_x'] = position[:,2:,0]
        traj['target_y'] = position[:,2:,1]

        return traj  
    def maze_env_open_walk(self,box_width,box_height,batch_size):
        linear_speed = 0.04
        n_step = 10
        #generate a sequence of nodes
        #set next node
        #if around next node, change next node, and turn head direction
        #detect if near wall, if yes, change head direction
        #generate random head turn, update head direction and position
        t_len = self.options.sequence_length
        #example one batch
        n_sample = self.options.sequence_length
        
        n_cell = t_len*linear_speed*batch_size/self.cell_len
        rs=np.random.randint(0,100000)
        traj = self.maze_randomwalk(steps=n_cell,random_seed=rs,mode='node')
        traj_pos = self.shift_node_pos[traj,:]
        diff_pos = np.diff(traj_pos,1,axis=0)
        n_diff_cell = (np.sum(np.abs(diff_pos),axis=1)/self.cell_len+0.00001).astype(int)
        
        #idx = np.nonzero(np.cumsum(np.sum(np.abs(diff_pos),axis=1))>(t_len*linear_speed*batch_size))
        #idx = idx[0][0]
        #traj = traj[:(idx+batch_size)]
        #traj_pos = traj_pos[:(idx+batch_size)]
        #n_diff_cell = n_diff_cell[:(idx+batch_size)]
        #import pdb;pdb.set_trace()
        idx_insert = np.nonzero(n_diff_cell>1)
        while len(idx_insert[0])>0:
          idx_insert = idx_insert[0][0]
          n_diff_cell_tmp = n_diff_cell[idx_insert]
          traj_pos = np.concatenate((traj_pos[:(idx_insert)],np.linspace(traj_pos[idx_insert],traj_pos[idx_insert+1],n_diff_cell_tmp+1),traj_pos[idx_insert+2:]))
          diff_pos = np.diff(traj_pos,1,axis=0)
          n_diff_cell = (np.sum(np.abs(diff_pos),axis=1)/self.cell_len+0.00001).astype(int)
          idx_insert = np.nonzero(n_diff_cell>1)
          
          
        
        tmp = np.insert(traj_pos,np.arange(1,len(traj_pos)-1,1),traj_pos[1:-1],axis=0)
        tmp = np.reshape(tmp,(2,len(traj_pos)-1,2),order='F')#start-end x batch x (x,y)
        
        tmp = np.linspace(tmp[0,:,:],tmp[1,:,:],n_step,endpoint=False)
        
        #then add random number
        idx = np.nonzero((tmp[-1,:,:] - tmp[0,:,:])==0)
        randn_drift = np.random.uniform(-self.corridor_width/2, self.corridor_width/2,size=(n_step,len(idx[0])))
        tmp[:,idx[0],idx[1]] = tmp[:,idx[0],idx[1]] + randn_drift

        n_page = int(np.ceil((t_len+2)/n_step))
        
        position = np.reshape(tmp[:,:(n_page*batch_size),:],(n_step*n_page,batch_size,2),order='F')
        position = position[:(t_len+2),:,:]
        tmp_vel = np.diff(position,n=1,axis=0)
        head_direction = np.zeros((position.shape[0],position.shape[1]))
        head_direction[1:,:] = np.arctan2(tmp_vel[:,:,1],tmp_vel[:,:,0])
        
        head_angular_velocity = np.zeros_like(head_direction)
        head_angular_velocity[1:,:] = np.diff(head_direction,n=1,axis=0)

        linear_velocity = np.zeros_like(head_direction)
        linear_velocity[1:,:] = np.linalg.norm(tmp_vel,axis=2)
         
        traj = {}
        # Input variables
        traj['init_hd'] = head_direction[0,:,None]
        traj['init_x'] = position[1,:,0,None]
        traj['init_y'] = position[1,:,1,None]

        traj['ego_v'] = np.squeeze(linear_velocity[2:,:]).T
        ang_v = head_angular_velocity
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:-1,:].T, np.sin(ang_v)[:-1,:].T

        # Target variables
        traj['target_hd'] = np.squeeze(head_direction[2:,:]).T
        traj['target_x'] = position[2:,:,0].T
        traj['target_y'] = position[2:,:,1].T
        return traj
     
    def maze_env_walk(self,box_width,box_height,batch_size):
      #print('hello world!')
        samples = self.options.sequence_length
        dt = 0.02  # time step increment (seconds)

        
        angular_speed = 30/180*np.pi
        linear_speed = self.options.box_height/50

        batch_linear_velocity = []#batch_size x T x output_neuron
        batch_position = []
        batch_head_angular_velocity = []
        batch_head_direction = []
        for i_batch in range(batch_size):
          current_head_direction = np.pi/2
          traj = self.maze_randomwalk(steps = 70, random_seed = np.random.randint(0,100000), mode = 'node')
          #print(traj,len(traj))

          head_direction = []
          head_angular_velocity = []
          position = []
          linear_velocity = []

          for i,node in enumerate(traj[0:-1]):
            current_node_pos = self.node_pos[node,:]
            next_node_pos = self.node_pos[traj[i+1],:]
            diff_pos = next_node_pos - current_node_pos
            
            tmp = diff_pos!=0
            if tmp[0] == tmp[1]:
              raise ValueError('traj has one node repeated!')
            next_node_direction = self.pos2polarangle(diff_pos)

            current_head_direction = next_node_direction

            #generate running through the corridor
            delta_t_linear = np.abs(np.sum(diff_pos)/linear_speed)
            int_delta_t = int(delta_t_linear)
            head_direction = head_direction + ((next_node_direction*np.ones((int_delta_t+2,1))).tolist())
            head_angular_velocity = head_angular_velocity + (np.zeros((int_delta_t+2,1)).tolist())
            
            tmp = np.zeros((int_delta_t+2))
            tmp[:int_delta_t+1] = np.ones(int_delta_t+1)*linear_speed
            tmp[0] = 0
            tmp[-1] = (delta_t_linear-int_delta_t)*linear_speed
            linear_velocity = linear_velocity+np.reshape(tmp,(-1,1)).tolist()

            tmp = np.sign(diff_pos)*np.repeat(tmp[:,None],1,axis=1)
            tmp_pos = np.reshape(current_node_pos,(1,2))+np.cumsum(tmp,axis=0)
            
            position = position+tmp_pos.tolist()
            
            

          linear_velocity = np.array(linear_velocity)
          position = np.array(position)
          head_angular_velocity = np.array(head_angular_velocity)
          head_direction = np.array(head_direction)

          batch_linear_velocity.append(linear_velocity)
          batch_position.append(position)
          batch_head_angular_velocity.append(head_angular_velocity)
          batch_head_direction.append(head_direction)

        t_len = self.options.sequence_length
  

        batch_linear_velocity = self.shrink_and_expand(batch_linear_velocity,t_len+2)
        batch_position = self.shrink_and_expand(batch_position,t_len+2)
        batch_head_angular_velocity = self.shrink_and_expand(batch_head_angular_velocity,t_len+2)
        batch_head_direction = self.shrink_and_expand(batch_head_direction,t_len+2)
        
        #linear_noise = np.random.normal(scale=linear_speed*0.01,size=batch_linear_velocity.shape)
        # angular_noise = np.random.normal(scale=angular_speed*0.01,size=batch_head_angular_velocity.shape)
        # noisy_input = np.concatenate((batch_linear_velocity+linear_noise,(batch_head_angular_velocity+angular_noise)),axis=2)
        # output = np.concatenate((batch_position,batch_head_direction),axis=2)

        #batch x sequence x feature
        traj = {}
        # Input variables
        traj['init_hd'] = batch_head_direction[:,0,None]
        traj['init_x'] = batch_position[:,1,0,None]
        traj['init_y'] = batch_position[:,1,1,None]

        traj['ego_v'] = np.squeeze(batch_linear_velocity[:,2:])
        ang_v = batch_head_angular_velocity
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        traj['target_hd'] = np.squeeze(batch_head_direction[:,2:])
        traj['target_x'] = batch_position[:,2:,0]
        traj['target_y'] = batch_position[:,2:,1]

        return traj
    def get_generator(self, batch_size=None, box_width=None, box_height=None):
        '''
        Returns a generator that yields batches of trajectories
        '''
        if not batch_size:
             batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height
            
        while True:
            traj = self.generate_trajectory(box_width, box_height, batch_size)
            
            v = np.stack([traj['ego_v']*np.cos(traj['target_hd']), 
                  traj['ego_v']*np.sin(traj['target_hd'])],axis=-1)
            v = torch.tensor(v,dtype=torch.float32).transpose(0,1)

            pos = np.stack([traj['target_x'], traj['target_y']],axis=-1)
            pos = torch.tensor(pos,dtype=torch.float32).transpose(0,1).cuda()
            place_outputs = self.place_cells.get_activation(pos)

            init_pos = np.stack([traj['init_x'], traj['init_y']],axis=-1)
            init_pos = torch.tensor(init_pos,dtype=torch.float32).cuda()
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            inputs = (v.cuda(), init_actv)
        
            yield (inputs, place_outputs, pos)



    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):
        ''' For testing performance, returns a batch of smample trajectories'''
        if not batch_size:
             batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height
            
        traj = self.generate_trajectory(box_width, box_height, batch_size)
        
        
        v = np.stack([traj['ego_v']*np.cos(traj['target_hd']), 
              traj['ego_v']*np.sin(traj['target_hd'])],axis=-1)
        v = torch.tensor(v,dtype=torch.float32).transpose(0,1)

        pos = np.stack([traj['target_x'], traj['target_y']],axis=-1)
        pos = torch.tensor(pos,dtype=torch.float32).transpose(0,1).cuda()
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj['init_x'], traj['init_y']],axis=-1)
        init_pos = torch.tensor(init_pos,dtype=torch.float32).cuda()
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        inputs = (v.cuda(), init_actv)
        
        return (inputs, place_outputs, pos)