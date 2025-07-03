close all; clear; clc;

if batchStartupOptionUsed
    return;
end
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

HETEROGENEOUS = true;

m = py.trimesh.load_mesh("./src/data/Model/eros_shape_200700.obj");
v = double(m.vertices) ./ 16;
f = uint32(m.faces) + 1;
patch('Faces', f, 'Vertices', v, 'FaceColor', [0.75 0.75 0.75], 'FaceAlpha', 0.67, 'EdgeColor', 'none');

if HETEROGENEOUS
    [x, y, z] = sphere(30);
    
    r         = nthroot(((m.volume/16e3)/10)*3/4/pi,3);
    hold on;
    surf(r*x-0.5, r*y, r*z, 'FaceColor', 'red', 'FaceAlpha', 1, 'EdgeColor', 'none');
    surf(r*x+0.5, r*y, r*z, 'FaceColor', 'blue', 'FaceAlpha', 1, 'EdgeColor', 'none');
end

axis equal image;
grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
xlim([-1.1 1.1]); ylim([-1.1 1.1]); zlim([-1.1 1.1]);
xticks(-1:0.5:1); yticks(-1:0.5:1); zticks(-1:0.5:1);
ax           = gca;
ax.Color     = [0.95 0.95 0.95];
ax.GridColor = [0.5 0.5 0.5];
ax.LineWidth = 1.5;

view(2);
camlight('left');
lighting gouraud;
material([0.3 0.8 0.1 20 1]);
view(-37.5, 45);

figname = "erosModel";
if HETEROGENEOUS
    figname = figname + "Heterogeneous";
else
    figname = figname + "Uniform";
end
exportgraphics(gcf, "./utils/" + figname + ".png", "Resolution", 300);