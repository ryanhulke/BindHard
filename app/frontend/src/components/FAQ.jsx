import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const FAQ_DATA = [
  {
    q: "What input data does Bind Hard need?",
    a: "A single PDB file containing your target protein pocket's structure. Upload it through the dashboard and the platform will match it against precomputed binding trajectories to identify candidate binders and their predicted 3D conformations.",
  },
  {
    q: "How does the binding prediction work?",
    a: "Bind Hard uses a flow matching model built on an equivariant graph neural network (EGNN). Given a target protein, the model generates candidate small-molecule binders by learning the distribution of binding conformations from a large dataset of known protein-ligand complexes.",
  },
  {
    q: "What does the trajectory viewer show?",
    a: "It renders the full binding trajectory — the sequence of 3D conformations the candidate binder explores as it docks into the target's binding site. You can scrub frame-by-frame or play it as an animation.",
  },
  {
    q: "How are candidates ranked?",
    a: "Candidate binders are sorted by their Vina docking score (lower is better), which estimates binding affinity in kcal/mol.",
  },
  {
    q: "Can I use this for drug discovery?",
    a: "Bind Hard is designed to accelerate the early-stage hit identification step. It helps researchers quickly explore which molecular scaffolds might bind a given target and visualize how they fit in the binding pocket. Hits should be followed up with wet-lab validation and more detailed simulations before drawing conclusions about clinical relevance.",
  },
];

function FaqItem({ q, a, index, isOpen, onToggle }) {
  return (
    <motion.div
      className="border-b border-white/8 transition-all duration-700"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={{ duration: 0.55, ease: "easeOut", delay: index * 0.06 }}
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between py-6 text-left group cursor-pointer"
        aria-expanded={isOpen}
      >
        <span className="text-white font-semibold text-base md:text-lg pr-8 group-hover:text-white transition-colors">
          {q}
        </span>
        <motion.span
          className="text-white text-2xl shrink-0"
          animate={{ rotate: isOpen ? 45 : 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          +
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {isOpen ? (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{
              height: { duration: 0.28, ease: "easeInOut" },
              opacity: { duration: 0.2, ease: "easeOut" },
            }}
            className="overflow-hidden"
          >
            <motion.p
              initial={{ y: -6 }}
              animate={{ y: 0 }}
              exit={{ y: -4 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              className="text-white text-sm leading-relaxed pb-6 pr-12"
            >
              {a}
            </motion.p>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.div>
  );
}

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState(0);

  return (
    <section
      id="faq"
      className="relative z-10 w-full max-w-4xl mx-auto px-6 pt-40 pb-32"
    >
      {/* Section heading */}
      <div className="text-center mb-16">
        <p className="text-white text-xs font-bold tracking-[0.3em] uppercase mb-4">
          Questions
        </p>
        <h2 className="font-eb-garamond text-white text-5xl md:text-7xl font-bold tracking-tight leading-[1.05]">
          FAQ
        </h2>
      </div>

      {/* FAQ list */}
      <div>
        {FAQ_DATA.map((item, i) => (
          <FaqItem
            key={i}
            q={item.q}
            a={item.a}
            index={i}
            isOpen={openIndex === i}
            onToggle={() => setOpenIndex((current) => (current === i ? -1 : i))}
          />
        ))}
      </div>
    </section>
  );
}
