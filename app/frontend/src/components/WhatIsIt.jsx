import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";

// const FEATURES = [
//   {
//     title: "Emotion Measurement",
//     desc: "Webcam-powered facial affect analysis tracks frustration, confusion, delight, and boredom at 10 Hz — so you know exactly what players feel, not just what they do.",
//   },
//   {
//     title: "DFA State Detection",
//     desc: "Gemini Vision watches gameplay video and maps every frame to your game's state graph. No SDK integration required — just record and go.",
//   },
//   {
//     title: "Physiological Sensing",
//     desc: "Apple Watch streams heart rate and HRV at 1 Hz, giving a second modality of arousal and stress independent of facial expression.",
//   },
//   {
//     title: "Intent vs Reality Verdicts",
//     desc: "Developers define the intended emotion per game state. PatchLab compares actual player affect to that intent and issues PASS / WARN / FAIL verdicts automatically.",
//   },
//   {
//     title: "Playtest Health Score",
//     desc: "A single 0–1 number that summarizes the entire session. Green means on-target, red means broken mechanics. Compare across testers instantly.",
//   },
//   {
//     title: "Sphinx AI Copilot",
//     desc: 'Ask questions about your playtest data in plain English — "Show average frustration per state as a heatmap" — and get instant SQL-backed visualizations.',
//   },
// ];

function FeatureCard({ icon, title, desc, delay }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ob = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          ob.disconnect();
        }
      },
      { threshold: 0.15 },
    );
    ob.observe(el);
    return () => ob.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="rounded-2xl border border-white/10 bg-white/[0.04] backdrop-blur-[20px] p-7 transition-all duration-700"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(32px)",
        transitionDelay: `${delay}ms`,
      }}
    >
      <span className="text-3xl mb-4 block">{icon}</span>
      <h3 className="text-white font-bold text-lg mb-2 tracking-tight">
        {title}
      </h3>
      <p className="text-white text-sm leading-relaxed">{desc}</p>
    </div>
  );
}

export default function WhatIsIt() {
  const pipelineCardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.55, ease: "easeOut" },
    },
  };

  const pipelineContainerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.14,
        delayChildren: 0.08,
      },
    },
  };

  return (
    <section
      id="what"
      className="relative z-10 w-full max-w-6xl mx-auto px-6 pt-48 pb-20"
    >
      {/* Section heading */}
      <div className="text-center mb-20">
        <p className="text-white text-xs font-bold tracking-[0.3em] uppercase mb-4">
          The Platform
        </p>
        <h2 className="font-eb-garamond text-white text-5xl md:text-7xl font-bold tracking-tight leading-[1.05]">
          What is it?
        </h2>
        <p className="mt-6 max-w-2xl mx-auto text-white text-lg leading-relaxed">
          Our protein binding tool is a generative AI framework that uses a flow
          matching model to predict and design how biomolecules
          bind in 3D space. Given a target such as a protein structure or
          sequence, it models possible binding interactions, estimates how
          molecules fit within binding sites, and can generate candidate binders
          for further screening. The goal is to help researchers generate high quality candidates and explore
          protein-ligand interactions faster and more efficiently in the early stages of drug discovery
        </p>
      </div>

      {/* Feature grid */}
      {/* <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {FEATURES.map((f, i) => (
          <FeatureCard key={f.title} {...f} delay={i * 80} />
        ))}
      </div> */}

      {/* Pipeline callout */}
      <div className="mt-16 rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-[20px] p-8 md:p-10">
        <h3 className="text-white font-bold text-lg mb-4 tracking-tight text-center">
          How it works
        </h3>
        <motion.div
          className="flex flex-col md:flex-row items-start md:items-center gap-4 md:gap-0"
          variants={pipelineContainerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.25 }}
        >
          {[
            {
              step: "01",
              label: "Upload Target",
              detail:
                "Start by uploading a protein pocket structure into the platform",
            },
            {
              step: "02",
              label: "Run the Model",
              detail:
                "A flow matching EGNN generates candidate small-molecule binders",
            },
            {
              step: "03",
              label: "Generate Output",
              detail:
                "The system exports structural files representing predicted molecular interactions and conformations",
            },
            {
              step: "04",
              label: "View and Analyze",
              detail:
                "Explore the predicted complex in an interactive interface for faster inspection",
            },
          ].map((s, i) => (
            <motion.div
              key={s.step}
              className="flex-1 flex items-start gap-3 md:flex-col md:items-center md:text-center"
              variants={pipelineCardVariants}
            >
              {i > 0 && (
                <div className="hidden md:block w-full h-px bg-gradient-to-r from-transparent via-white/15 to-transparent -mt-3 mb-3" />
              )}
              <span className="text-white font-mono text-xs font-bold">
                {s.step}
              </span>
              <div>
                <p className="text-white font-semibold text-sm">{s.label}</p>
                <p className="text-white text-xs mt-0.5 leading-relaxed">
                  {s.detail}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
