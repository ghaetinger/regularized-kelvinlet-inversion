#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texmap;

out vec2 TexMap3D;

void main()
{
   TexMap3D = texmap;
   gl_Position = position;//vec4(position, 0.0f, 1.0f);
};
