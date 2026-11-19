/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Minimal Vite/TypeScript build-harness stub.
 * The canonical interactive research workbench is the 10-page Streamlit app
 * served via `start_dev.sh` (`app/streamlit_app.py`) on port 3000.
 */
export default function App() {
  return (
    <main className="min-h-screen bg-[#F8F9FA] text-[#202124] flex items-center justify-center p-6 font-sans">
      <div className="max-w-lg bg-white border border-[#DADCE0] rounded-lg p-6">
        <h1 className="text-lg font-medium text-[#202124]">
          Gemini 4 Hydrogen Cross-Modal Safety Research Workbench
        </h1>
        <p className="mt-2 text-sm text-[#5F6368]">
          The canonical interface for this study is the 10-page Python Streamlit workbench
          (<code>app/streamlit_app.py</code>), launched via <code>bash start_dev.sh</code>.
        </p>
      </div>
    </main>
  );
}
