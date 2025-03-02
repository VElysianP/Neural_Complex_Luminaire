
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

// NOTE(yaoyi): these two files are added for the network inference
#include <mitsuba/complexlum/ComplexLum_Map_Sphere.h>
#include <time.h>
#include <omp.h>
// #include <mitsuba/LightFieldNetEigen.h>

using namespace nvinfer1;
using namespace std;
MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

// NOTE(yaoyi): for eval network

ICudaEngine* lf_createCudaEngine(string const& onnxModelPath, int batchSize)
{
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{ nvinfer1::createInferBuilder(gLogger) };
	unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{ builder->createNetworkV2(explicitBatch) };
	unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{ nvonnxparser::createParser(*network, gLogger) };
	unique_ptr<nvinfer1::IBuilderConfig, Destroy<nvinfer1::IBuilderConfig>> config{ builder->createBuilderConfig() };

	if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
	{
		cout << "ERROR: could not parse input engine." << endl;
		return nullptr;
	}

	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	builder->setFp16Mode(builder->platformHasFastFp16());
	builder->setMaxBatchSize(batchSize);

	auto profile = builder->createOptimizationProfile();
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims2{ 1, 6 });
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims2{ 200000, 6});
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims2{ 200000 , 6 });
	config->addOptimizationProfile(profile);

	return builder->buildEngineWithConfig(*network, *config);
}


// NOTE(yaoyi): for sampling weight network
ICudaEngine* sw_createCudaEngine(string const& onnxModelPath, int batchSize)
{
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{ nvinfer1::createInferBuilder(gLogger) };
	unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{ builder->createNetworkV2(explicitBatch) };
	unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{ nvonnxparser::createParser(*network, gLogger) };
	unique_ptr<nvinfer1::IBuilderConfig, Destroy<nvinfer1::IBuilderConfig>> config{ builder->createBuilderConfig() };

	if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
	{
		cout << "ERROR: could not parse input engine." << endl;
		return nullptr;
	}

	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	//builder->setFp16Mode(builder->platformHasFastFp16());
	builder->setMaxBatchSize(batchSize);

	auto profile = builder->createOptimizationProfile();
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims2{ 1, 3 });
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims2{ 400000, 3 });
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims2{ 400000, 3 });
	config->addOptimizationProfile(profile);

	return builder->buildEngineWithConfig(*network, *config);
}



ICudaEngine* alpha_createCudaEngine(string const& onnxModelPath, int batchSize)
{
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{ nvinfer1::createInferBuilder(gLogger) };
	unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{ builder->createNetworkV2(explicitBatch) };
	unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{ nvonnxparser::createParser(*network, gLogger) };
	unique_ptr<nvinfer1::IBuilderConfig, Destroy<nvinfer1::IBuilderConfig>> config{ builder->createBuilderConfig() };

	if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
	{
		cout << "ERROR: could not parse input engine." << endl;
		return nullptr;
	}

	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
	//builder->setFp16Mode(builder->platformHasFastFp16());
	builder->setMaxBatchSize(batchSize);

	auto profile = builder->createOptimizationProfile();
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims2{ 1, 6 });
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims2{ 200000, 6 });
	profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims2{ 200000, 6 });
	config->addOptimizationProfile(profile);

	return builder->buildEngineWithConfig(*network, *config);
}



class NetworkMIS : public MonteCarloIntegrator {
public:
	NetworkMIS(const Properties &props)
		: MonteCarloIntegrator(props) {
		// NOTE(yaoyi): for initialize the network Eigen version
		// eigen_net.initialize("total_weights.bin", 373.6657, 2770.0388, 0);

		// NOTE(yaoyi): for initalize the network TensorRT version
		// this will only happen in this integrator

		// NOTE(yaoyi): for eval network
		lf_netstruct_file = props.getString("LFfile");
		lf_batchSize = props.getInteger("LFbatchSize");
		// for sampling weight network
		sw_netstruct_file = props.getString("SWfile");
		sw_batchSize = props.getInteger("SWbatchSize");

		alpha_netstruct_file = props.getString("Alphafile");
		alpha_batchSize = props.getInteger("AlphabatchSize");

		// pdf_netstruct_file = props.getString("PDFfile");
		// pdf_batchSize = props.getInteger("PDFbatchSize");

		luminaire_Center = props.getPoint("lumCenter");
		luminaire_bound_radius = props.getFloat("lumBoundRadius");
		SLog(EInfo, "radiuse: %f", luminaire_bound_radius);
		SLog(EInfo, "center: %f,%f,%f", luminaire_Center.x, luminaire_Center.y, luminaire_Center.z);
		bounding_sphere_radius = luminaire_bound_radius;
		bounding_sphere_center = luminaire_Center;
		boundingType = props.getString("boundingType","sphereNative");
		boundLocal2World = props.getTransform("local2world");

		// init eval inference
		lf_net_batchSize = lf_batchSize;
		lf_engine.reset(lf_createCudaEngine(lf_netstruct_file, lf_batchSize));
		if (!lf_engine) return;
		assert(lf_engine->getNbBindings() == 2);
		assert(lf_engine->bindingIsInput(0) ^ lf_engine->bindingIsInput(1));

		// init sampling weight inference
		sw_net_batchSize = sw_batchSize;
		sw_engine.reset(sw_createCudaEngine(sw_netstruct_file, sw_batchSize));
		if (!sw_engine) return;


		alpha_net_batchSize = alpha_batchSize;
		alpha_engine.reset(alpha_createCudaEngine(alpha_netstruct_file, alpha_batchSize));
		if (!alpha_engine) return;
		assert(alpha_engine->getNbBindings() == 2);
		assert(alpha_engine->bindingIsInput(0) ^ sw_engine->bindingIsInput(1));

		// init pdf network
		// pdf_net_batchSize = pdf_batchSize;
		// pdf_engine.reset(pdf_createCudaEngine(pdf_netstruct_file, pdf_batchSize));
		// if (!pdf_engine) return;
		// assert(pdf_engine->getNbBindings() == 2);
		// assert(pdf_engine->bindingIsInput(0) ^ pdf_engine->bindingIsInput(1));
	}

	/// Unserialize from a binary data stream
	NetworkMIS(Stream *stream, InstanceManager *manager)
		: MonteCarloIntegrator(stream, manager) {
	}

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec, LightField_Inf& inf_data) {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(r);
		Spectrum Li(0.0f);
		bool scattered = false;

		/* Perform the first ray intersection (or ignore if the
		intersection has already been provided). */
		rRec.rayIntersect(ray);
		ray.mint = Epsilon;

		Spectrum throughput(1.0f);
		Float eta = 1.0f;

		// NOTE(yaoyi: bounce start from 1)
		// but bounce it actually useless in current structure
		int bounce = 0;

		while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
			bounce++;
			if (!its.isValid()) {
				/* If no intersection could be found, potentially return
				radiance from a environment luminaire if it exists */
				if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
					&& (!m_hideEmitters || scattered))
					// NOTE(yaoyi): here I ignored the env light
					// Li += throughput * scene->evalEnvironment(ray);
					break;
			}

			const BSDF *bsdf = its.getBSDF(ray);

			/* Possibly include emitted radiance if requested */
			// NOTE(yaoyi): do we need to consider light field query here?
			// I think I can avoid considering this part since it will not bring about huge difference
			// but on the other hand, I should also consider the data transfer between CPU and GPU 
			if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
				&& (!m_hideEmitters || scattered))
			{
				// Li += throughput * its.Le(-ray.d);
				// NOTE(yaoyi): lf replacement 1: record the uvtp values for inference
				LightField_Inf_Inner lf_infer_data_temp1;
				LightField_Inf_Inner alpha_infer_data_temp1;
				// TODO(yaoyi): - or no - (used to have -)
				// also transform into local space
//				SLog(EInfo, "radiuse: %f", luminaire_bound_radius);
	//			SLog(EInfo, "position: %f,%f,%f", its.p.x, its.p.y, its.p.z);
				Vector4f uvtp_temp1 = FromPosDir2uvtp(boundLocal2World, its, luminaire_Center, luminaire_bound_radius, -ray.d, boundingType);
				Vector nor_position;
				nor_position = (its.p - luminaire_Center);
				nor_position = nor_position / luminaire_bound_radius;
				nor_position = nor_position + Vector(1.f);
				nor_position = nor_position / 2.f;
				Vector nor_dir = (-ray.d);
				lf_infer_data_temp1.posx = nor_position[0];
				lf_infer_data_temp1.posy = nor_position[1];
				lf_infer_data_temp1.posz = nor_position[2];
				lf_infer_data_temp1.dirx = nor_dir[0];
				lf_infer_data_temp1.diry = nor_dir[1];
				lf_infer_data_temp1.dirz = nor_dir[2];
				lf_infer_data_temp1.isdirect = true;
				alpha_infer_data_temp1.posx = nor_position[0];
				alpha_infer_data_temp1.posy = nor_position[1];
				alpha_infer_data_temp1.posz = nor_position[2];
				alpha_infer_data_temp1.dirx = nor_dir[0];
				alpha_infer_data_temp1.diry = nor_dir[1];
				alpha_infer_data_temp1.dirz = nor_dir[2];
				alpha_infer_data_temp1.isdirect = true;
				//lf_infer_data_temp1.theta = 0.5;
				//lf_infer_data_temp1.phi = 0.5;
		//		SLog(EInfo, "tp: %f,%f", uvtp_temp1[2], uvtp_temp1[3]);
				lf_infer_data_temp1.throughput = throughput;
				inf_data.lf_inner.push_back(lf_infer_data_temp1);
				inf_data.alpha_inner.push_back(alpha_infer_data_temp1);
				bool flag = true;
				lf_infer_data_temp1.isblack = false;
				while (flag)
				{
					ray = Ray(its.p, ray.d, ray.time);
				//	SLog(EInfo, "its.p%f",its.p[0]);
					if (scene->rayIntersect(ray, its))
					{
					//	SLog(EInfo, "its222.p%f", its.p[0]);
						if (its.isEmitter())
						{
							flag = false;
							if (!scene->rayIntersect(ray, its))
							{
								
								lf_infer_data_temp1.isblack = true;
								flag = false;
							}
						}
						else
						{
							lf_infer_data_temp1.isblack = true;
						}
					}
					else
						flag = false;
				}
				rRec.depth--;
			}

			/* Include radiance from a subsurface scattering model if requested */
			if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
				// WARN(yaoyi): the Subsurface scattering was disabled because of the code structure
				// Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

				if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
					|| (m_strictNormals && dot(ray.d, its.geoFrame.n)
						* Frame::cosTheta(its.wi) >= 0)) {

					/* Only continue if:
					1. The current path length is below the specifed maximum
					2. If 'strictNormals'=true, when the geometric and shading
					normals classify the incident direction to the same side */
					break;
				}

			/* ==================================================================== */
			/*                     Direct illumination sampling                     */
			/* ==================================================================== */

			/* Estimate the direct illumination if this is requested */
			DirectSamplingRecord dRec(its);

			if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
				(bsdf->getType() & BSDF::ESmooth)) {

				// NOTE(yaoyi): replace here for the sampling weight network
				Point2 ksi12 = rRec.nextSample2D();
				// Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
				Spectrum value = scene->networkSampleEmitterDirect(dRec, ksi12);

				Point fakecenter = luminaire_Center;
				fakecenter.x += 0;
				fakecenter.y += 0;
				fakecenter.z += 0;
				const Vector CenterToRef = dRec.ref - fakecenter;
			//	SLog(EInfo, "pos: %f,%f,%f", dRec.ref.x, dRec.ref.y, dRec.ref.z);
				const float refDist = CenterToRef.length();
				Vector CTR = normalize(CenterToRef);
			//	Vector CTR = normalize(CenterToRef);
				float xx = CenterToRef.x / luminaire_bound_radius;
				float yy = CenterToRef.y / luminaire_bound_radius;
				float zz = CenterToRef.z / luminaire_bound_radius;
				if ((xx*xx + yy * yy+zz*zz) > 18*18)
				{
					float scale = 18.f / sqrt(xx*xx + yy * yy+zz*zz);
					xx *= scale;
					yy *= scale;
					zz *= scale;
				}
				xx /= 42.f;
				xx = xx + 0.5;			
				yy /= 42.f;
				yy = yy + 0.5;				
				zz /= 42.f;
				zz = zz + 0.5;
				float scale = 1.0;
			
			//	SLog(EInfo, "pos: %f,%f,%f", xx, yy, zz);
				float theta = acos(CTR.z);
				theta /= M_PI;

				float phi = std::atan2(CTR.z, CTR.x);
				if (phi < 0.f) phi += M_PI * 2.0;
				phi /= (M_PI * 2.0);
				//				phi = 1 - phi;
				float RA = refDist / luminaire_bound_radius;
				RA /= 21.f; //this is an importance normalization 
							// END
							//			SLog(EInfo, "RA��%f", RA);
							// NOTE(yaoyi): record the sampling weight inference value
				SamplingWeight_Inf_Inner sw_inf_data_temp1;
				//!!!!!!!!!!!!!!!!!!!!!!!!!ZJQ:z-up version please change from here!
				sw_inf_data_temp1.theta = xx
					;
				sw_inf_data_temp1.phi = yy ;
				sw_inf_data_temp1.radius = zz ;
				sw_inf_data_temp1.ksi1 = ksi12[0];
				sw_inf_data_temp1.ksi2 = ksi12[1];

				sw_inf_data_temp1.directIts = its;
				sw_inf_data_temp1.directBSDF = bsdf;
				sw_inf_data_temp1.throughput = throughput;
				inf_data.sw_inner.push_back(sw_inf_data_temp1);

			}

			/* ==================================================================== */
			/*                            BSDF sampling                             */
			/* ==================================================================== */

			/* Sample BSDF * cos(theta) */
			Float bsdfPdf;
			BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
			Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
			if (bsdfWeight.isZero())
				break;

			scattered |= bRec.sampledType != BSDF::ENull;

			/* Prevent light leaks due to the use of shading normals */
			const Vector wo = its.toWorld(bRec.wo);
			Float woDotGeoN = dot(its.geoFrame.n, wo);
			if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
				break;

			bool hitEmitter = false;
			Spectrum value;
			bool hitEnv = false;
			// NOTE(yaoyi): this is for BSDF network query
			LightField_Inf_Inner lf_infer_data_temp2;
			LightField_Inf_Inner alpha_infer_data_temp2;
			/* Trace a ray in this direction */
			ray = Ray(its.p, wo, ray.time);
			if (scene->rayIntersect(ray, its)) {
				/* Intersected something - check if it was a luminaire */
				if (its.isEmitter()) {
					// value = its.Le(-ray.d);

					// NOTE(yaoyi): lf replacement 2-1
					// Vector loca_dir = its.toLocal(-ray.d);
					Vector4f uvtp_temp2 = FromPosDir2uvtp(boundLocal2World, its, luminaire_Center, luminaire_bound_radius, -ray.d, boundingType);
					Vector nor_position;
					nor_position = (its.p - luminaire_Center);
					nor_position = nor_position / luminaire_bound_radius;
					nor_position = nor_position + Vector(1.f);
					nor_position = nor_position / 2.f;
					Vector nor_dir = (-ray.d);
					lf_infer_data_temp2.posx = nor_position[0];
					lf_infer_data_temp2.posy = nor_position[1];
					lf_infer_data_temp2.posz = nor_position[2];
					lf_infer_data_temp2.dirx = nor_dir[0];
					lf_infer_data_temp2.diry = nor_dir[1];
					lf_infer_data_temp2.dirz = nor_dir[2];
					alpha_infer_data_temp2.posx = nor_position[0];
					alpha_infer_data_temp2.posy = nor_position[1];
					alpha_infer_data_temp2.posz = nor_position[2];
					alpha_infer_data_temp2.dirx = nor_dir[0];
					alpha_infer_data_temp2.diry = nor_dir[1];
					alpha_infer_data_temp2.dirz = nor_dir[2];					
					lf_infer_data_temp2.isdirect = false;
					lf_infer_data_temp2.collectindex = bounce-1;
					lf_infer_data_temp2.bsdfpdf = bsdfPdf;					
					alpha_infer_data_temp2.isdirect = false;
					alpha_infer_data_temp2.collectindex = bounce-1;
					dRec.setQuery(ray, its);
					hitEmitter = true;
				}
			}
			else {
				/* Intersected nothing -- perhaps there is an environment map? */
				const Emitter *env = scene->getEnvironmentEmitter();

				if (env) {
					if (m_hideEmitters && !scattered)
						break;

					value = env->evalEnvironment(ray);
					if (!env->fillDirectSamplingRecord(dRec, ray))
						break;
					hitEmitter = true;
					hitEnv = true;
				}
				else {
					break;
				}
			}

			/* Keep track of the throughput and relative
			refractive index along the path */
			throughput *= bsdfWeight;
			eta *= bRec.eta;

			/* If a luminaire was hit, estimate the local illumination and
			weight using the power heuristic */
			if (hitEmitter &&
				(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
				/* Compute the prob. of generating that direction using the
				implemented direct illumination sampling technique */
				const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
					scene->pdfEmitterDirect(dRec) : 0;
				// Li += throughput * value * miWeight(bsdfPdf, lumPdf);

				// lf replacement 2-1
				if (hitEnv)
					Li += throughput * value * miWeight(bsdfPdf, lumPdf);
				else
				{
					//NOTE(yaoyi): once pdf network is added, should add the miWeight here
					lf_infer_data_temp2.throughput = throughput;// * miWeight(bsdfPdf, lumPdf);
					inf_data.lf_inner.push_back(lf_infer_data_temp2);
					inf_data.alpha_inner.push_back(alpha_infer_data_temp2);
					float flag = true;
					lf_infer_data_temp2.isblack = false;
					while (flag)
					{
						ray = Ray(its.p, ray.d, ray.time);
						if (scene->rayIntersect(ray, its))
						{
							if (its.isEmitter())
								flag = false;
							else
							{
								lf_infer_data_temp2.isblack = true;
							}
						}
						else
							flag = false;
					}
				//	inf_data.lf_inner.push_back(lf_infer_data_temp2);// NOTE(yaoyi): push back the inferece data to the vector
				}
			}

			/* ==================================================================== */
			/*                         Indirect illumination                        */
			/* ==================================================================== */

			/* Set the recursive query type. Stop if no surface was hit by the
			BSDF sample or if indirect illumination was not requested */
			if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
				break;
			rRec.type = RadianceQueryRecord::ERadianceNoEmission;

			if (rRec.depth++ >= m_rrDepth) {
				/* Russian roulette: try to keep path weights equal to one,
				while accounting for the solid angle compression at refractive
				index boundaries. Stop with at least some probability to avoid
				getting stuck (e.g. due to total internal reflection) */

				Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
				if (rRec.nextSample1D() >= q)
					break;
				throughput /= q;
			}
		}

		/* Store statistics */
		avgPathLength.incrementBase();
		avgPathLength += rRec.depth;

		return Li;
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA;
		pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		MonteCarloIntegrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "NetworkMIS[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

private:
	// LightFieldCompressionNet eigen_net;
	std::string lf_netstruct_file;
	int lf_batchSize;

	std::string sw_netstruct_file;
	int sw_batchSize;
	std::string alpha_netstruct_file;
	int alpha_batchSize;

	std::string boundingType;
	Point luminaire_Center;
	float luminaire_bound_radius;
	Transform boundLocal2World;
	MTS_DECLARE_CLASS()

};

MTS_IMPLEMENT_CLASS_S(NetworkMIS, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(NetworkMIS, "Network Inference integrator with MIS");
MTS_NAMESPACE_END
