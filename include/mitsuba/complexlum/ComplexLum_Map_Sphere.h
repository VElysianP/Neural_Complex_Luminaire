#pragma once

#include "mitsuba.h"
#include "render/shape.h"
MTS_NAMESPACE_BEGIN

// Point mCenter = Point(0.0,0.0,0.0);
// Point mCenter = Point(383.18, 34.342, -0.119);
// float boundingRadius = 42.f;
// Point mCenter = Point(86.425, -29.089, 40.503);
// float boundingRadius = 13.5f;

//need to make sure that uvtp are all within (0,1)
void Fromuvtp2PosDir(Point mCenter, float boundingRadius, Vector4f uvtp, Point & position, Vector & direction, std::string boundingtype)
{
    //read data
    float u = uvtp[0];
    float v = uvtp[1];
    float theta = uvtp[2];
    float phi = uvtp[3];

    //calculate for pos
    float z = (1 - 2 * u)*boundingRadius;
    // float z = (2 * u - 1.f)*boundingRadius;
    float xy = math::safe_sqrt(boundingRadius*boundingRadius - z*z);
    z += mCenter[2];
    float x = xy*cos(v * 2 * M_PI);
    x += mCenter[0];
    float y = xy*sin(v * 2 * M_PI);
    y += mCenter[1];
    position[0]=x;position[1]=y;position[2]=z;

    //calculate for dir
    float dirz = (1 - 2 * theta);
    // float dirz = 2 * theta - 1.f;
    float dirxy = static_cast<float>(math::safe_sqrt(1.0 - dirz * dirz));
    float dirx = dirxy*cos(phi * 2 * M_PI);
    float diry = dirxy*sin(phi * 2 * M_PI);
    direction[0]=dirx;direction[1]=diry;direction[2]=dirz;
    
}

Vector4f sphereNative_FromPosDir2uvtp(Point mCenter, float boundingRadius, Point position, Vector direction)
{
    float us, vs;
    Vector SP = position - mCenter;
    Vector bounding = normalize(SP);
    us = 1 - bounding.z;
    us /= 2;
    vs = std::atan2(bounding.y, bounding.x);
    if (vs < float(0)) vs += M_PI * 2.0;
    vs /= (M_PI * 2.0);

    Vector ID = normalize(direction);
    float theta = 1 - ID.z;
    theta /= 2;

    // float theta = std::acos(ID.z);
    // theta /= (M_PI * 2.0);
    float phi = std::atan2(ID.y, ID.x);
    if (phi < float(0)) phi += M_PI * 2.0;
    phi /= (M_PI * 2.0);

    return Vector4f(us, vs, theta, phi);
}

Vector4f sphereRevised_FromPosDir2uvtp(Point mCenter, float boundingRadius, Point position, Vector direction)
{
    float us, vs;
	Vector SP = position - mCenter;
	Vector bounding = normalize(SP);
	us = 1.f - bounding[1];
	us /= 2.f;
	vs = std::atan2(bounding[2], bounding[0]);
	if (vs < float(0)) vs += static_cast<float>(M_PI * 2.f);
	vs /= static_cast<float>(M_PI * 2.f);
	Vector ID = normalize(direction);
	float theta = 1.f - ID[1];
	theta /= 2.f;
	
	float phi = std::atan2(ID[2], ID[0]);
	if (phi < float(0)) phi += static_cast<float>(M_PI * 2.f);
	phi /= static_cast<float>(M_PI * 2.f);
	
	return Vector4f(us, vs, theta, phi);
}

mitsuba::Vector4f cube_FromPosDir2uvtp(mitsuba::Transform& transform, mitsuba::Intersection& its, mitsuba::Vector direction)
{
    return mitsuba::Vector4f(0.f);
}

mitsuba::Vector4f sphere_FromPosDir2uvtp(mitsuba::Transform& transform, mitsuba::Intersection& its, mitsuba::Vector direction)
{
    return mitsuba::Vector4f(0.f);
}

mitsuba::Vector4f cylinder_FromPosDir2uvtp(mitsuba::Transform& transform, mitsuba::Intersection& its, mitsuba::Vector direction)
{
    mitsuba::Transform world2LocalTransform = transform.inverse();
    mitsuba::Point localPoint = world2LocalTransform(its.p);

    // calculate for position
    float u, v, theta, phi;
    // top
    if (localPoint.y - 1.0 < 1e-6)
    {
        u = localPoint.x * 0.25f + 0.25f;
        v = localPoint.z * 0.25f + 0.75f;
    }
    //bottom
    else if (localPoint.y + 1.0 < 1e-6)
    {
        u = localPoint.x * 0.25f + 0.75f;
        v = localPoint.z * 0.25f + 0.75f;
    }
    else
    {
        float phi_radius = std::atan2(localPoint.z, localPoint.x);
        if (phi_radius < 0)
            phi_radius += 2 * M_PI;
        u = phi_radius / (2 * M_PI);

        v = (localPoint.y + 1.f) / 0.5f;
    }

    // calculate for direction: using tangent space
    // NOTE(yaoyi): this direction is assumed to be the ray direction
    // therefore it has a - here 
    mitsuba::Vector localNormal = its.geoFrame.n;
    mitsuba::Vector normalizedDir = mitsuba::normalize(direction);

    theta = mitsuba::dot(localNormal, -normalizedDir) / (localNormal.length() * normalizedDir.length());
    // TODO(how should I deal with reversed direction)
    if(theta < 0.f || theta > M_PI * 0.5f) theta = 0.f;
    theta /= (0.5f * M_PI);

    Vector stProjectedDir = normalizedDir + localNormal * cos(theta);
    phi = mitsuba::dot(its.geoFrame.s, stProjectedDir) / (stProjectedDir.length() * its.geoFrame.s.length());
    if (mitsuba::dot(its.geoFrame.t, stProjectedDir) < 0.f)
    {
        phi += M_PI;
    }
    phi /= (2.f * M_PI);

    return mitsuba::Vector4f(u, v, theta, phi);
}

mitsuba::Vector4f FromPosDir2uvtp(mitsuba::Transform& transform, Intersection& its, mitsuba::Point mCenter, float boundRadius, mitsuba::Vector direction, std::string boundingtype)
{
    if (boundingtype == "sphereNative")
    {
        return sphereNative_FromPosDir2uvtp(mCenter, boundRadius, its.p, direction);
    }

    if (boundingtype == "sphereRevised")
    {
        return sphereRevised_FromPosDir2uvtp(mCenter, boundRadius, its.p, direction);
    }

    if (boundingtype == "cylinder")
    {
        return cylinder_FromPosDir2uvtp(transform, its, direction);
    }

    if (boundingtype == "cube")
    {

    }
}

// TODO(yaoyi): need to revise this part
// this is not universal now
Vector2 FromShadingPoint2tp(Point mCenter, float boundRadius, Point shadingPoint)
{
// THIS IS THE PREVIOUS VERSION
    // Vector dir = mitsuba::normalize(shadingPoint - mCenter);
    // // Point lum_its_point = mCenter + dir * boundRadius;
    // // Vector4 tempuvtp = sphereNative_FromPosDir2uvtp(mCenter, boundRadius, lum_its_point, Vector(1.0));
    // // return Vector2(tempuvtp[0], tempuvtp[1]);

    // float theta = (1 - dir.z) / 2;
    // float phi = std::atan2(dir.y, dir.x);
    // if (phi < 0.0)
    //     phi += M_PI * 2.0;
    // phi /= (M_PI * 2.0);
    // return Vector2(theta, phi);
// END

// THE NEW VERSION
    Vector dir = mitsuba::normalize(shadingPoint - mCenter);

    float theta = (1 - dir.y) / 2;
    float phi = std::atan2(dir.z, dir.x);
    if (phi < 0.0)
        phi += M_PI * 2.0;
    phi /= (M_PI * 2.0);
    return Vector2(theta, phi);
// END
}

MTS_NAMESPACE_END