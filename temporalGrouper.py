
class TemporalGrouper:

    def __init__(
        self,
        max_gap_seconds: float = 2.0,
        max_duration_seconds: float = 4.0
    ):

        self.max_gap_seconds = max_gap_seconds
        self.max_duration_seconds = max_duration_seconds

    def group_frames(self, frames: list[dict]) -> list[dict]:

        if not frames:
            return []

        # Sort retrieved frames by their global timestamps
        sorted_frames = sorted(
            frames,
            key=lambda frame: frame["metadata"]["global_timestamp"]
        )

        groups = []
        current_group = [sorted_frames[0]]

        # Group frames using both temporal constraints
        for frame in sorted_frames[1:]:

            previous_frame = current_group[-1]

            previous_timestamp = (
                previous_frame["metadata"]["global_timestamp"]
            )

            current_timestamp = (
                frame["metadata"]["global_timestamp"]
            )

            first_timestamp = (
                current_group[0]["metadata"]["global_timestamp"]
            )

            time_gap = current_timestamp - previous_timestamp
            group_duration = current_timestamp - first_timestamp

            # Check both consecutive gap and total group duration
            if (
                time_gap <= self.max_gap_seconds
                and group_duration <= self.max_duration_seconds
            ):

                current_group.append(frame)

            else:
                groups.append(current_group)
                current_group = [frame]

        # Add the final group
        groups.append(current_group)

        # Convert frame groups into candidate moments
        moments = []

        for group in groups:

            # Select the most relevant frame as representative
            representative = max(
                group,
                key=lambda frame: frame["similarity"]
            )

            timestamps = [
                frame["metadata"]["global_timestamp"]
                for frame in group
            ]

            # Calculate frame similarity statistics
            similarities = [
                frame["similarity"] for frame in group
            ]

            average_similarity = (
                sum(similarities) / len(similarities)
            )

            max_similarity = max(similarities)

            # Calculate combined moment relevance score
            moment_score = (
                0.7 * max_similarity
                + 0.3 * average_similarity
            )

            moments.append({
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration": max(timestamps) - min(timestamps),
                "frame_count": len(group),
                "average_similarity": average_similarity,
                "max_similarity": max_similarity,
                "moment_score": moment_score,
                "representative_frame": representative,
                "frames": group
            })

        # Rank moments using combined relevance score
        moments.sort(
            key=lambda moment: moment["moment_score"],
            reverse=True
        )

        return moments

    def display_moments(self, moments: list[dict], top_k: int = 5):

        if not moments:
            print("\nNo matching moments found.")
            return

        selected_moments = moments[:top_k]

        print(f"\nFound {len(selected_moments)} candidate moments:")
        print("=" * 65)

        for rank, moment in enumerate(selected_moments, start=1):

            representative = moment["representative_frame"]
            metadata = representative["metadata"]

            print(f"\nMoment #{rank}")

            print(
                f"Time range: "
                f"{moment['start_time']:.2f}s - "
                f"{moment['end_time']:.2f}s"
            )

            print(f"Chunk: {metadata['chunk_id']}")
            print(f"Frames grouped: {moment['frame_count']}")

            print(
                f"Average similarity: "
                f"{moment['average_similarity']:.4f}"
            )

            print(
                f"Maximum similarity: "
                f"{moment['max_similarity']:.4f}"
            )

            print(
                f"Moment relevance score: "
                f"{moment['moment_score']:.4f}"
            )

            print(
                f"Representative frame: "
                f"{metadata['frame_id']}"
            )

            print(
                f"Description: "
                f"{representative['description']}"
            )

            print("-" * 65)
