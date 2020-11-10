#version 330 core

in vec2 TexMap3D;

uniform float timestep;
uniform float a;
uniform float b;
uniform float c;
uniform vec3 pos;
uniform vec3 force;
uniform float radius;
uniform vec3 textprop;
uniform int fixBorder;

float val;
vec3 vals; 

layout(location = 0) out vec4 color;

uniform sampler3D text3d;

float PI = 3.1415926535897932384626433832795;
float Pi = 3.1415926535897932384626433832795;
int fixRadius = 50;

float Power(float v, float w)
{
    return pow(v, w);
}

mat3 productWithTranspost(vec3 x)
{
  return mat3(x[0] * x[0], x[1] * x[0], x[2] * x[0],
              x[0] * x[1], x[1] * x[1], x[2] * x[1],
              x[0] * x[2], x[1] * x[2], x[2] * x[2]);
}

vec3 fixInPosition(vec3 p, float inflRadius)
{
    vec3 outFix = vec3(0.0f, 0.0f, 0.0f);

    float dx1 = p[0] - 1;
    float dx2 = textprop[0] - p[0];
    float dy1 = p[1] - 1;
    float dy2 = textprop[1] - p[1];
    float dz1 = p[2] - 1;
    float dz2 = textprop[2] - p[2];

    float dx = min(dx1, dx2);
    float dy = min(dy1, dy2);
    float dz = min(dz1, dz2);

    outFix[0] = sin(PI * (min(dx, inflRadius) / inflRadius)/2);
    outFix[1] = sin(PI * (min(dy, inflRadius) / inflRadius)/2);
    outFix[2] = sin(PI * (min(dz, inflRadius) / inflRadius)/2);

    return outFix;
}

vec3 getForce(vec3 point, bool shouldfix)
{
    vec3 r = point - pos;
    float rLength = length(r);
    float rEpslon = sqrt(pow(radius, 2) + pow(rLength, 2));
    mat3 I = mat3(1.0f);
    mat3 first = float((a - b)/rEpslon) * I;
    mat3 second = float(b/pow(rEpslon, 3)) * productWithTranspost(r);
    mat3 third = float((a/2) * ((pow(radius, 2)/pow(rEpslon, 3)))) * I;
    mat3 kelvinState = first + second + third;
    vec3 u = c * radius * kelvinState * force;
    vec3 fix = fixInPosition(point, fixRadius);

    if (shouldfix) {
		u = u * fix;
    }
    return u;
}

mat3 getJacobianMtx(vec3 p, bool fix)
{
    float x = p[0];
    float y = p[1];
    float t = p[2];
    float ox = pos[0];
    float oy = pos[1];
    float ot = pos[2];
    float z = radius;
    float fx = force[0];
    float fy = force[1];
    float ft = force[2];

    float J11 = c*z*((3*b*ft*(-ot + t)*(ox - x)*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fy*(ox - x)*(-ox + x)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*ft*(-ot + t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (b*fy*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fx*((3*b*(ox - x)*Power(-ox + x,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ox - x)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ox - x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (2*b*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J21 = c*z*((3*b*ft*(-ot + t)*(ox - x)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fx*(ox - x)*(-ox + x)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fx*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fy*((3*b*(ox - x)*Power(-oy + y,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ox - x)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ox - x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J31 = c*z*((3*b*fx*(-ot + t)*(ox - x)*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fy*(-ot + t)*(ox - x)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fx*(-ot + t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + ft*((3*b*Power(-ot + t,2)*(ox - x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ox - x)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ox - x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J12 = c*z*((3*b*ft*(-ot + t)*(-ox + x)*(oy - y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fy*(-ox + x)*(oy - y)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fy*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fx*((3*b*Power(-ox + x,2)*(oy - y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(oy - y)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(oy - y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J22 = c*z*((3*b*ft*(-ot + t)*(oy - y)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fx*(-ox + x)*(oy - y)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*ft*(-ot + t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (b*fx*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fy*((3*b*(oy - y)*Power(-oy + y,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(oy - y)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(oy - y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (2*b*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J32 = c*z*((3*b*ft*(-ot + t)*(oy - y)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fx*(-ox + x)*(oy - y)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*ft*(-ot + t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (b*fx*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fy*((3*b*(oy - y)*Power(-oy + y,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(oy - y)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(oy - y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (2*b*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J13 = c*z*((3*b*ft*(ot - t)*(-ot + t)*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fy*(ot - t)*(-ox + x)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*ft*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fx*((3*b*(ot - t)*Power(-ox + x,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ot - t)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ot - t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J23 = c*z*((3*b*ft*(ot - t)*(-ot + t)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fx*(ot - t)*(-ox + x)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*ft*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + fy*((3*b*(ot - t)*Power(-oy + y,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ot - t)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ot - t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));
    float J33 = c*z*((3*b*fx*(ot - t)*(-ot + t)*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*b*fy*(ot - t)*(-ot + t)*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (b*fx*(-ox + x))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (b*fy*(-oy + y))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + ft*((3*b*(ot - t)*Power(-ot + t,2))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5) + (3*a*(ot - t)*Power(z,2))/(2.*Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),2.5)) + ((a - b)*(ot - t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5) + (2*b*(-ot + t))/Power(Power(ot - t,2) + Power(ox - x,2) + Power(oy - y,2) + Power(z,2),1.5)));

    if (fix) {
        float sx = textprop[0];
        float sy = textprop[1];
        float st = textprop[2];

        int varx = 0;
        int vary = 0;
        int vart = 0;

        if (sx - 2 * x <= -1 && sx - x - fixRadius <= 0) {
            varx = -1;
        } else if (sx - 2 * x > -1 && x - fixRadius <= 1) {
            varx = 1;
        }

        if (sy - 2 * y <= -1 && sy - y - fixRadius <= 0) {
            vary = -1;
        } else if (sy - 2 * y > -1 && y - fixRadius <= 1) {
            vary = 1;
        }

        if (st - 2 * t <= -1 && st - t - fixRadius <= 0) {
            vart = -1;
        } else if (st - 2 * t > -1 && t - fixRadius <= 1) {
            vart = 1;
        }


        vec3 u = getForce(p, false);
        float ux = u[0];
        float uy = u[1];
        float ut = u[2];

        vec3 B = fixInPosition(p, fixRadius);
        float Bx = B[0];
        float By = B[1];
        float Bt = B[2];

        float xborderDiff = (Pi*cos((Pi*min(sx - x,min(-1 + x,fixRadius)))/(2*fixRadius))*varx)/(2. * fixRadius);
        float yborderDiff = (Pi*cos((Pi*min(sy - y,min(-1 + y,fixRadius)))/(2*fixRadius))*vary)/(2. * fixRadius);
        float tborderDiff = (Pi*cos((Pi*min(st - t,min(-1 + t,fixRadius)))/(2*fixRadius))*vart)/(2. * fixRadius);

		vec3 db = vec3(xborderDiff, yborderDiff, tborderDiff);

		vec3 newDiag = u * db;

		mat3 newMtx = mat3(newDiag[0], 0, 0,
				0, newDiag[1], 0,
				0, 0, newDiag[2]);

		mat3 toSum = mat3(Bx, 0, 0,
				0, By, 0,
				0, 0, Bt);

		mat3 J = mat3(J11, J21, J31,
				J12, J22, J32,
				J13, J23, J33);

		toSum = toSum * J; 

		return newMtx + toSum;
        }

    return mat3(J11, J21, J31,
			J12, J22, J32,
			J13, J23, J33);
}

vec3 jacobToMiplevel(mat3 J, float t)
{
    float iLvl = 1 + t * abs(min(0, J[0][0]));
    float jLvl = 1 + t * abs(min(0, J[1][1]));
    float tLvl = 1 + t * abs(min(0, J[2][2]));

    return vec3(iLvl, jLvl, tLvl);
}

vec3 solve(mat3 A, vec3 bvec)
{
    return inverse(A) * bvec;
}

vec3 point_inversion(vec3 q, int maxIter, float GNDamping, float threshold)
{
    vec3 p = vec3(q);
    int i = 0;
    mat3 I = mat3(1.0f);
    while (i < maxIter) {
        mat3 J = getJacobianMtx(p, bool(fixBorder));
        vec3 K = getForce(p, bool(fixBorder));
        vec3 residue = p + K - q;
        vec3 bvec = (transpose(J) + I) * residue;
        mat3 A = transpose(J)*J + 2.0f * transpose(J) + I;
        vec3 delta = solve(A, bvec);
        p = p - GNDamping*delta;
        if (length(residue) < threshold) {
                break;
        }
        i = i + 1;
    }
    return p;
}

float retardationFunction(float alpha)
{
    float ret = 1.0f/(pow(2.0f, (1.0f/4.0f))) * pow((cos(PI * alpha) + 1.0f), (1.0f/4.0f));
    return ret;
}

vec3 rayleigh_eigen_one(mat3 A, vec3 initial_guess)
{
    int max_iterations = 5;
    vec3 vec = initial_guess;
    val = dot(vec, A * vec) / dot(vec, vec);
	int i = 0;
	mat3 I = mat3(1.0f);
	while (i < max_iterations) {
        vec = solve(A - val*I, vec);
        vec = vec / length(vec);
        val = dot(vec, A * vec) / dot(vec, vec);
		i++;
	}
    return vec;
}

vec3 rayleigh_eigen_3x3(mat3 A)
{
    vec3 vec0 = rayleigh_eigen_one(A, vec3(1,0,0));
	float val0 = val;
    vec3 rotated_vec0 = vec3(-1 * vec0[2], vec0[0], vec0[2]);
    vec3 vec1 = rayleigh_eigen_one(A, rotated_vec0);
	float val1 = val;
    vec3 vec_2 = cross(vec0, vec1);
    float val2 = dot(vec_2, A * vec_2) / dot(vec_2, vec_2);
	return vec3(val0, val1, val2);
    /*
	vals = [val0, val, val2]
    vecs = hcat(vec0, vec1, vec2)
    p = sortperm(vals)
    vals = vals[p]
	*/
}

vec3 computeJacobianEigenValues(mat3 A)
{
	float p1 = pow(A[0][1], 2) + pow(A[0][2], 2) + pow(A[1][2], 2);
	float eig1;
	float eig2;
	float eig3;
	float q;
	float p2;
	float p;
	float r;
	mat3 B;
	float phi;

	mat3 I = mat3(1.0f);

	if (p1 == 0) { 
		eig1 = A[0][0];
		eig2 = A[1][1];
		eig3 = A[2][2];
	} else {
		q = (A[0][0] + A[1][1] + A[2][2])/3;
		p2 = pow(A[0][0] - q, 2) + pow(A[1][1] - q, 2) + pow(A[2][2] - q, 2) + 2 * p1;
		p = sqrt(p2 / 6);
		B = (1 / p) * (A - q * I);
		r = determinant(B) / 2;

		if (r <= -1) {
			phi = PI / 3;
		} else if (r >= 1) {
			phi = 0;
		} else {
			phi = acos(r) / 3;
		}
	}

   eig1 = q + 2 * p * cos(phi);
   eig3 = q + 2 * p * cos(phi + (2*PI/3));
   eig2 = 3 * q - eig1 - eig3;

   return vec3(eig1, eig2, eig3);
}

void main()
{
    vec3 q = vec3(textprop[0] * TexMap3D[0],
                  textprop[1] * TexMap3D[1],
                  timestep);

    vec3 pinv = point_inversion(q, 500, 0.5f, 0.1f);

    vec3 p = vec3(pinv[0] / textprop[0],
                  pinv[1] / textprop[1],
                  pinv[2] / textprop[2]);


    color = texture(text3d, p);

	mat3 J = getJacobianMtx(pinv, bool(fixBorder));
	vec3 eigvals = rayleigh_eigen_3x3((J + transpose(J))/2);
	if (min(eigvals[0], min(eigvals[1], eigvals[2])) < -1) {
		color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	}
};
