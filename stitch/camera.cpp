#include <iostream>
#include "camera.h"

namespace surround360 {

	const Camera::Real Camera::kNearInfinity = 1e6;

	void Camera::setRotation(
		const Vector3& forward,
		const Vector3& up,
		const Vector3& right) {
//		CHECK_LT(right.cross(up).dot(forward), 0) << "rotation must be right-handed";
		rotation.row(2) = -forward; // +z is back
		rotation.row(1) = up; // +y is up
		rotation.row(0) = right; // +x is right
								 // re-unitarize
		const Camera::Real tol = 0.001;
//		CHECK(rotation.isUnitary(tol)) << rotation << " is not close to unitary";
		Eigen::AngleAxis<Camera::Real> aa(rotation);
		rotation = aa.toRotationMatrix();
	}

	void Camera::setRotation(const Vector3& forward, const Vector3& up) {
		setRotation(forward, up, forward.cross(up));
	}

	Camera::Camera(const Type type, const Vector2& res, const Vector2& focal) :
		type(type), resolution(res), focal(focal) {
		position.setZero();
		rotation.setIdentity();
		principal = resolution / 2;
		distortion.setZero();
		setDefaultFov();
	}

	Camera::Camera(const dynamic& json) {
//		CHECK_GE(json["version"].asDouble(), 1.0);

		id = json["id"].getString();

		type = deserializeType(json["type"]);

		position = deserializeVector<3>(json["origin"]);

		setRotation(
			deserializeVector<3>(json["forward"]),
			deserializeVector<3>(json["up"]),
			deserializeVector<3>(json["right"]));

		resolution = deserializeVector<2>(json["resolution"]);

		if (json.count("principal")) {
			principal = deserializeVector<2>(json["principal"]);
		}
		else {
			principal = resolution / 2;
		}

		if (json.count("distortion")) {
			distortion = deserializeVector<2>(json["distortion"]);
		}
		else {
			distortion.setZero();
		}

		if (json.count("fov")) {
			setFov(json["fov"].asDouble());
		}
		else {
			setDefaultFov();
		}

		focal = deserializeVector<2>(json["focal"]);

		if (json.count("group")) {
			group = json["group"].getString();
		}
	}

#ifndef SUPPRESS_RIG_IO

	dynamic Camera::serialize() const {
		dynamic result = dynamic::object
		("version", 1)
			("type", serializeType(type))
			("origin", serializeVector(position))
			("forward", serializeVector(forward()))
			("up", serializeVector(up()))
			("right", serializeVector(right()))
			("resolution", serializeVector(resolution))
			("principal", serializeVector(principal))
			("focal", serializeVector(focal))
			("id", id);
		if (!distortion.isZero()) {
			result["distortion"] = serializeVector(distortion);
		}
		if (!isDefaultFov()) {
			result["fov"] = getFov();
		}
		if (!group.empty()) {
			result["group"] = group;
		}

		return result;
	}

#endif // SUPPRESS_RIG_IO

	void Camera::setRotation(const Vector3& angleAxis) {
		// convert angle * axis to rotation matrix
		Real angle = angleAxis.norm();
		Vector3 axis = angleAxis / angle;
		if (angle == 0) {
			axis = Vector3::UnitX();
		}
		rotation = Eigen::AngleAxis<Real>(angle, axis).toRotationMatrix();
	}

	Camera::Vector3 Camera::getRotation() const {
		// convert rotation matrix to angle * axis
		Eigen::AngleAxis<Real> angleAxis(rotation);
		if (angleAxis.angle() > M_PI) {
			angleAxis.angle() = 2 * M_PI - angleAxis.angle();
			angleAxis.axis() = -angleAxis.axis();
		}

		return angleAxis.angle() * angleAxis.axis();
	}

	void Camera::setScalarFocal(const Real& scalar) {
		focal = { scalar, -scalar };
	}

	Camera::Real Camera::getScalarFocal() const {
		//CHECK_EQ(focal.x(), -focal.y()) << "pixels are not square";
		return focal.x();
	}

	void Camera::setFov(const Real& fov) {
		//CHECK(fov <= M_PI / 2 || type == Type::FTHETA);
		Real cosFov = std::cos(fov);
		fovThreshold = cosFov * std::abs(cosFov);
	}

	Camera::Real Camera::getFov() const {
		return fovThreshold < 0
			? std::acos(-std::sqrt(-fovThreshold))
			: std::acos(std::sqrt(fovThreshold));
	}

	void Camera::setDefaultFov() {
		if (type == Type::FTHETA) {
			fovThreshold = -1;
		}
		else {
			//CHECK(type == Type::RECTILINEAR) << "unexpected: " << int(type);
			fovThreshold = 0;
		}
	}

	bool Camera::isDefaultFov() const {
		return type == Type::FTHETA ? fovThreshold == -1 : fovThreshold == 0;
	}

	Camera::Real cross2(const Camera::Vector2& a, const Camera::Vector2& b) {
		return -a.y() * b.x() + a.x() * b.y();
	}

	Camera::Vector3 midpoint(
		const Camera::Ray& a,
		const Camera::Ray& b,
		const bool forceInFront) {
		// this is the mid-point method

		// find ta and tb that minimizes the distance between
		// a(ta) = pa + ta * va and b(tb) = pb + tb * vb:

		// then return the mid-point between a(ta) and b(tb)

		// d(a - b)^2/dta = 0 &&
		// d(a - b)^2/dtb = 0 <=>
		// dot( va, 2 * (a(ta) - b(tb))) = 0 &&
		// dot(-vb, 2 * (a(ta) - b(tb))) = 0 <=>
		// dot(va, a(ta) - b(tb)) = 0 &&
		// dot(vb, a(ta) - b(tb)) = 0 <=>
		// dot(va, pa) + ta * dot(va, va) - dot(va, pb) - tb * dot(va, vb) = 0 &&
		// dot(vb, pa) + ta * dot(vb, va) - dot(vb, pb) - tb * dot(vb, vb) = 0 <=>

		// reformulate as vectors
		//    fa * ta - fb * tb + fc = (0, 0), where
		//    m = rows(va, vb)
		//    fa = m * va
		//    fb = m * vb
		//    fc = m * (pa - pb)
		// -det(fa, fb) * ta + det(fb, fc) = 0 &&
		// -det(fa, fb) * tb + det(fa, fc) = 0 <=>
		// ta = det(fb, fc) / det(fa, fb) &&
		// tb = det(fa, fc) / det(fa, fb)
		Eigen::Matrix<Camera::Real, 2, 3> m;
		m.row(0) = a.direction();
		m.row(1) = b.direction();
		Camera::Vector2 fa = m * a.direction();
		Camera::Vector2 fb = m * b.direction();
		Camera::Vector2 fc = m * (a.origin() - b.origin());
		Camera::Real det = cross2(fa, fb);
		Camera::Real ta = cross2(fb, fc) / det;
		Camera::Real tb = cross2(fa, fc) / det;

		// check for parallel lines
		if (!std::isfinite(ta) || !std::isfinite(tb)) {
			ta = tb = Camera::kNearInfinity;
		}

		// check whether intersection is behind camera
		if (forceInFront && (ta < 0 || tb < 0)) {
			ta = tb = Camera::kNearInfinity;
		}

		Camera::Vector3 pa = a.pointAt(ta);
		Camera::Vector3 pb = b.pointAt(tb);
		return (pa + pb) / 2;
	}

	Camera::Rig Camera::loadRig(const std::string& filename) {
		boost::property_tree::ptree tree;
		boost::property_tree::json_parser::read_json(std::ifstream(filename), tree);

		Camera::Rig rig;
		for (const auto& camera : tree.get_child("cameras")) {
			rig.emplace_back(camera.second);
		}
		return rig;
	}

} // namespace surround360

