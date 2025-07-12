close all; clear; clc;

if batchStartupOptionUsed
    return;
end
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

m = py.trimesh.load_mesh("./src/data/Model/eros_shape_200700.obj");
v = double(m.vertices) ./ 16;
f = uint32(m.faces) + 1;
patch('Faces', f, 'Vertices', v, 'FaceColor', [0.75 0.75 0.75], 'FaceAlpha', 1, 'EdgeColor', 'none');

% ----------
hold on;
field = load("./src/preprocessing/metricsData.mat");
idx = 0==field.mPlanesTRJ(:,3:3);
trj = field.mPlanesTRJ(idx, 1:2);
acc = field.mPlanesACC(idx, 1:2);
pot = field.mPlanesPOT(idx);

[vx,vy] = deal(v(:,1),v(:,2));
shp     = alphaShape(vx, vy);

[x,y,u,v,p] = deal(trj(:,1),trj(:,2),acc(:,1),acc(:,2),pot);
[xlin,ylin] = deal(linspace(-2,2,50));
[Xg, Yg]    = meshgrid(xlin,ylin);
[Ug,Vg,Pg]  = deal(griddata(x,y,u,Xg,Yg,"linear"),griddata(x,y,v,Xg,Yg,"linear"),griddata(x,y,p,Xg,Yg,"linear"));

inShape = inShape(shp, Xg, Yg);
[Ug(inShape),Vg(inShape),Pg(inShape)] = deal(NaN);

contour(Xg,Yg,Pg,10,"LineWidth",1,"Color",[149,149,249]./255)
quiver(Xg,Yg,Ug,Vg,"r","AutoScaleFactor",0.7,"LineWidth",2,"Color",[224,49,49]./255);

% ----------

axis equal image;
grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
xlim([-1.5 1.5]); ylim([-1.5 1.5]); zlim([-1.5 1.5]);
xticks(0:0:0); yticks(0:0:0); zticks(0:0:0);
ax           = gca;
ax.Color     = [0.95 0.95 0.95];
ax.GridColor = [0.5 0.5 0.5];
ax.LineWidth = 1.5;

view(2);
camlight('left');
lighting gouraud;
material([0.3 0.8 0.1 20 1]);

exportgraphics(gcf, "./utils/erosModelField.png", "Resolution", 300);