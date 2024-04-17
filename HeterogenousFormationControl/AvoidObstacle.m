function [dstatedt] = AvoidObstacle(t,state,a,b,omega)

% Constants
alpha = 10;
t1 = 15;
k = 10; % chosen by me

% Connectivity/ Interagent Communication
B = 1/2*[1 1 1];
A = [1 1/4 1/4;
     1/2 1 0;
     1/2 0 1];

% Positions/ Formations
etadv1 = [1/(1+exp(-alpha*(t-t1))) -1/(1+exp(alpha*(t-t1)))]';
etadv2 = [2/(1+exp(-alpha*(t-t1))) 1/(1+exp(alpha*(t-t1)))]';
etadv3 = [1/(1+exp(-alpha*(t-t1))) -1/(1+exp(alpha*(t-t1))) -1-1/(1+exp(-alpha*(t-t1)))]';
etad12 = [1/(1+exp(-alpha*(t-t1))) 2/(1+exp(alpha*(t-t1)))]';
etad21 = [-1/(1+exp(-alpha*(t-t1))) -2/(1+exp(alpha*(t-t1)))]';
etad13 = [0 0 -1-1/(1+exp(-alpha*(t-t1)))]';
etad31 = [0 0]';
epsdvi = 0;
epsdii = 0;
epsdotd = 0;

% state vector
xv = state(1);
yv = state(2);
xdotv = state(3);
ydotv = state(4);
x1 = state(5);
y1 = state(6);
x2 = state(7);
y2 = state(8);
x3 = state(9);
y3 = state(10);
z3 = state(11);
xdot1 = state(12);
ydot1 = state(13);
xdot2 = state(14);
ydot2 = state(15);
xdot3 = state(16);
ydot3 = state(17);
zdot3 = state(18);

% Equations of Motion

dxv = xdotv;
dyv = ydotv;
dxdotv = -omega^2 * xv * (a^2 / (a^2 + b^2));
dydotv = -omega^2 * yv * (b^2 / (a^2 + b^2));
dx1 = xdot1;
dy1 = ydot1;
dx2 = xdot2;
dy2 = ydot2;
dx3 = xdot3;
dy3 = ydot3;
dz3 = zdot3;
dxdot1 = epsdotd - k*(x1 - B(1)*(xv - etadv1(1)) - A(1,2)*(x2-etad21(1)) - A(1,3)*(x3-etad31(1))) ...
    - k*(xdot1 - B(1)*(xdotv - epsdvi) - A(1,2)*(xdot2-epsdii) - A(1,3)*(xdot3-epsdii));
dydot1 = epsdotd - k*(y1 - B(1)*(yv - etadv1(2)) - A(1,2)*(y2-etad21(2)) - A(1,3)*(y3-etad31(2))) ...
    - k*(ydot1 - B(1)*(ydotv - epsdvi) - A(1,2)*(ydot2-epsdii) - A(1,3)*(ydot3-epsdii));
dxdot2 = epsdotd - k*(x2 - B(2)*(xv - etadv2(1)) - A(2,1)*(x1-etad12(1))) ...
    - k*(xdot2 - B(2)*(xdotv - epsdvi) - A(2,1)*(xdot1-epsdii));
dydot2 = epsdotd - k*(y2 - B(2)*(yv - etadv2(2)) - A(2,1)*(y1-etad12(2))) ...
    - k*(ydot2 - B(2)*(ydotv - epsdvi) - A(2,1)*(ydot1-epsdii));
dxdot3 = epsdotd - k*(x3 - B(3)*(xv - etadv3(1)) - A(3,1)*(x1-etad13(1))) ...
    - k*(xdot3 - B(3)*(xdotv - epsdvi) - A(3,1)*(xdot1-epsdii));
dydot3 = epsdotd - k*(y3 - B(3)*(yv - etadv3(2)) - A(3,1)*(y1-etad13(2))) ...
    - k*(ydot3 - B(3)*(ydotv - epsdvi) - A(3,1)*(ydot1-epsdii));
dzdot3 = epsdotd - k*(z3 - B(3)*(0 - etadv3(3)) - A(3,1)*(0-etad13(3))) ...
    - k*(zdot3 - B(3)*(0 - epsdvi) - A(3,1)*(0-epsdii));

% Pack into output
dstatedt = [dxv; dyv; dxdotv; dydotv; dx1; dy1; dx2; dy2; dx3; dy3; dz3; ...
    dxdot1; dydot1; dxdot2; dydot2; dxdot3; dydot3; dzdot3;];


