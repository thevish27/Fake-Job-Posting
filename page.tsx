import Navbar from "@/components/Navbar";
import DetectionForm from "@/components/DetectionForm";

export default function DetectPage() {
  return (
    <>
      <Navbar />

      <main className="max-w-3xl mx-auto px-6 py-20">
        <div className="bg-white/80 backdrop-blur-lg border border-gray-200 rounded-2xl p-10 shadow-xl">

          {/* Header Section */}
          <div className="text-center mb-12">
            <h1 className="text-3xl font-bold mb-4">
              Fake Job Detection
            </h1>

            <p className="text-gray-600">
              Analyze job postings using AI-powered fraud detection.
            </p>
          </div>

          {/* Form */}
          <DetectionForm />
        </div>
      </main>
    </>
  );
}
