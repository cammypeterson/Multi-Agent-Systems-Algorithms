%% Time-varying Formation Control
% This code reconstructs the results from the Formation
% Control paper by Rahimi, et. al.
clear
clc
close all

% UAV and UGV parameters
Mq = 0.53;          % kg
Ix = 0.1676;
Iy = 0.1686;
Iz = 0.2974;
Lq = 0.5;           % m
g = 9.81;           % m/s^2
Mr = 23;            % kg
Lr = 0.5;           % m
Jr = 1;


%% Example 1 - Obstacle Swarming
omega = 0.125;
a = 6;
b = 7;

% Initial Positions (close to formation positions but slightly off)
Pv = [a,0];
Vv = [0,-b/2*omega];
P1 = [Pv(1)-1 Pv(2)+0.5];
P2 = [Pv(1)+1 Pv(2)-2];
P3 = [Pv(1)-0.5 Pv(2)+1.5 0];
V1 = [0 0];
V2 = [0 0];
V3 = [0 0 0];

% Set up ODE45
z0 = [Pv Vv P1 P2 P3 V1 V2 V3];
tspan = [0 30];

u_func = @(t,z)AvoidObstacle(t,z,a,b,omega);
[tout,zout] = ode45(u_func,tspan,z0);

% Access output from ODE45
xv = zout(:,1);
yv = zout(:,2);
x1 = zout(:,5);
y1 = zout(:,6);
x2 = zout(:,7);
y2 = zout(:,8);
x3 = zout(:,9);
y3 = zout(:,10);
z3 = zout(:,11);

% Plot XY
figure(1)
plot(xv, yv,':', x1, y1,'--', x2, y2,'--', x3, y3,'-.','Linewidth',1.5);
legend('Virtual Leader','UGV1','UGV2','UAV1','Location','northwest')
xlabel('x (m)');
ylabel('y (m)');
axis equal; % Make the aspect ratio equal to see the oval shape
grid on;

% Plot Z
figure(2)
plot(tout,z3);
xlabel('t (s)');
ylabel('z (m)');
grid on;


%% Example 2 - Coverage

% Initial Positions (close to formation positions but slightly off)
Pv = [0,0];
Vv = [0.25,0.25];
P1 = [Pv(1)+2 Pv(2)+0.5];
P2 = [Pv(1)-1 Pv(2)+1.5];
P3 = [Pv(1)-1 Pv(2)+0];
P4 = [Pv(1)+1 Pv(2)-2 0];
V1 = [0 0];
V2 = [0 0];
V3 = [0 0];
V4 = [0 0 0];

% Set up ODE45
z0 = [Pv Vv P1 P2 P3 P4 V1 V2 V3 V4];
tspan = [0 30];

u_func = @(t,z)ExpandCoverage(t,z);
[tout,zout] = ode45(u_func,tspan,z0);

% Access output from ODE45
xv = zout(:,1);
yv = zout(:,2);
x1 = zout(:,5);
y1 = zout(:,6);
x2 = zout(:,7);
y2 = zout(:,8);
x3 = zout(:,9);
y3 = zout(:,10);
x4 = zout(:,11);
y4 = zout(:,12);
z4 = zout(:,13);

% Plot XY
figure(3)
plot(xv, yv,':', x1, y1,'--', x2, y2,'--', x3, y3,'--',x4,y4,'-.','Linewidth',1.5);
legend('Virtual Leader','UGV1','UGV2','UGV3','UAV1','Location','southeast')
xlabel('x (m)');
ylabel('y (m)');
axis equal;
grid on;