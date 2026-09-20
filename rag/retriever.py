import pickle
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


class GovernmentRetriever:

    def __init__(
        self,
        vectorstore_path,
        embedding_model
    ):

        vectorstore_path = Path(
            vectorstore_path
        )

        # ----------------------------------------------------
        # LOAD FAISS
        # ----------------------------------------------------

        self.index = faiss.read_index(
            str(
                vectorstore_path /
                "faiss.index"
            )
        )

        # ----------------------------------------------------
        # LOAD METADATA
        # ----------------------------------------------------

        with open(
            vectorstore_path /
            "metadata.pkl",
            "rb"
        ) as f:

            self.metadata = pickle.load(
                f
            )

        # ----------------------------------------------------
        # VALIDATE FAISS + METADATA
        # ----------------------------------------------------

        if self.index.ntotal != len(
            self.metadata
        ):

            raise ValueError(
                "FAISS index and metadata "
                "are not synchronized.\n"
                f"FAISS vectors: {self.index.ntotal}\n"
                f"Metadata records: {len(self.metadata)}"
            )

        # ----------------------------------------------------
        # LOAD EMBEDDING MODEL
        # ----------------------------------------------------

        self.embedding_model = (
            SentenceTransformer(
                embedding_model
            )
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=6,
        department=None
    ):

        # ----------------------------------------------------
        # 1. QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )


        # ----------------------------------------------------
        # 2. SEARCH MORE CANDIDATES
        # ----------------------------------------------------
        #
        # We retrieve more candidates than the UI asks for.
        #
        # Example:
        #
        # User asks for 6
        #
        # FAISS searches 18
        #
        # This gives us more useful neighboring chunks.
        # ----------------------------------------------------

        candidate_k = min(
            max(top_k * 3, top_k),
            self.index.ntotal
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                candidate_k
            )
        )


        # ----------------------------------------------------
        # 3. PREPARE RESULTS
        # ----------------------------------------------------

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            item = (
                self.metadata[idx]
                .copy()
            )

            item["score"] = float(
                score
            )

            results.append(
                item
            )


        # ----------------------------------------------------
        # 4. DEPARTMENT FILTER
        # ----------------------------------------------------

        if department:

            results = [
                item
                for item in results

                if item["department"].strip().lower()
                ==
                department.strip().lower()
            ]


        # ----------------------------------------------------
        # 5. SORT BY DOCUMENT ORDER
        # ----------------------------------------------------
        #
        # FAISS gives similarity order.
        #
        # We now restore:
        #
        # source
        # page
        # chunk_on_page
        #
        # ----------------------------------------------------

        results.sort(
            key=lambda item: (
                item.get(
                    "source",
                    ""
                ),

                item.get(
                    "page",
                    0
                ),

                item.get(
                    "chunk_on_page",
                    0
                )
            )
        )


        # ----------------------------------------------------
        # 6. LIMIT RESULTS
        # ----------------------------------------------------

        results = results[
            :top_k
        ]


        return results