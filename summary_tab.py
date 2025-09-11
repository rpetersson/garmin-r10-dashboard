# Summary tab content
with tab0:
    st.subheader("📊 Performance Summary")
    
    # Overall performance metrics in a grid
    summary_cols = st.columns(3)
    
    # Distance Performance
    with summary_cols[0]:
        st.markdown("### 🎯 **Distance Performance**")
        if 'Carry Distance' in club_df.columns and club_df['Carry Distance'].notna().any():
            avg_carry = club_df['Carry Distance'].mean()
            max_carry = club_df['Carry Distance'].max()
            std_carry = club_df['Carry Distance'].std()
            
            st.metric("Average Carry", f"{avg_carry:.1f}m")
            st.metric("Best Shot", f"{max_carry:.1f}m")
            st.metric("Consistency (±)", f"{std_carry:.1f}m")
            
            # Distance rating
            if 'Driver' in title_suffix:
                if avg_carry > 250: rating = "Excellent 🔥"
                elif avg_carry > 220: rating = "Good 👍"
                elif avg_carry > 180: rating = "Average 📊"
                else: rating = "Needs Work 💪"
            elif any(iron in title_suffix for iron in ['7 Iron']):
                if avg_carry > 160: rating = "Excellent 🔥"
                elif avg_carry > 140: rating = "Good 👍"
                elif avg_carry > 120: rating = "Average 📊"
                else: rating = "Needs Work 💪"
            else:
                rating = "Data Available 📊"
            
            st.caption(f"Rating: {rating}")
    
    # Efficiency Performance  
    with summary_cols[1]:
        st.markdown("### ⚡ **Efficiency**")
        if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
            avg_smash = club_df['Smash Factor'].mean()
            optimal_shots = club_df[(club_df['Smash Factor'] >= 1.25) & (club_df['Smash Factor'] <= 1.35)]
            efficiency_pct = (len(optimal_shots) / len(club_df)) * 100
            
            st.metric("Avg Smash Factor", f"{avg_smash:.3f}")
            st.metric("Optimal Range %", f"{efficiency_pct:.1f}%")
            
            if avg_smash >= 1.28: rating = "Excellent 🔥"
            elif avg_smash >= 1.22: rating = "Good 👍"
            elif avg_smash >= 1.15: rating = "Average 📊"
            else: rating = "Needs Work 💪"
            
            st.caption(f"Rating: {rating}")
        
        if 'Ball Speed' in club_df.columns and club_df['Ball Speed'].notna().any():
            avg_ball_speed = club_df['Ball Speed'].mean()
            st.metric("Avg Ball Speed", f"{avg_ball_speed:.1f} km/h")
    
    # Accuracy Performance
    with summary_cols[2]:
        st.markdown("### 🎯 **Accuracy**")
        if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
            direction_std = club_df['Launch Direction'].std()
            straight_shots = club_df[club_df['Launch Direction'].abs() <= 5]
            accuracy_pct = (len(straight_shots) / len(club_df)) * 100
            
            st.metric("Direction Spread", f"±{direction_std:.1f}°")
            st.metric("Straight Shots %", f"{accuracy_pct:.1f}%")
            
            if direction_std <= 8: rating = "Excellent 🔥"
            elif direction_std <= 12: rating = "Good 👍"
            elif direction_std <= 18: rating = "Average 📊"
            else: rating = "Needs Work 💪"
            
            st.caption(f"Rating: {rating}")
        
        if 'Sidespin' in club_df.columns and club_df['Sidespin'].notna().any():
            avg_sidespin = club_df['Sidespin'].abs().mean()
            st.metric("Avg Sidespin", f"{avg_sidespin:.0f} rpm")
    
    # Detailed Recommendations
    st.subheader("🎯 Personalized Recommendations")
    
    recommendations = []
    priority_areas = []
    
    # Smash Factor Analysis
    if 'Smash Factor' in club_df.columns and club_df['Smash Factor'].notna().any():
        avg_smash = club_df['Smash Factor'].mean()
        if avg_smash < 1.20:
            priority_areas.append("Smash Factor")
            recommendations.append({
                "icon": "⚡",
                "title": "Improve Smash Factor (Priority)",
                "issue": f"Your average smash factor is {avg_smash:.3f}, below optimal range",
                "solution": "Focus on center face contact. Practice with impact tape or use alignment sticks for consistent setup.",
                "target": "Target: 1.25-1.35 range"
            })
        elif avg_smash < 1.25:
            recommendations.append({
                "icon": "⚡",
                "title": "Optimize Smash Factor",
                "issue": f"Your smash factor of {avg_smash:.3f} has room for improvement",
                "solution": "Work on consistent ball striking. Check your grip pressure and swing tempo.",
                "target": "Target: 1.30+ for maximum efficiency"
            })
    
    # Distance Consistency
    if 'Carry Distance' in club_df.columns and club_df['Carry Distance'].notna().any():
        distance_std = club_df['Carry Distance'].std()
        if distance_std > 15:  # High variance in distance
            priority_areas.append("Distance Control")
            recommendations.append({
                "icon": "📏",
                "title": "Improve Distance Control",
                "issue": f"Distance spread of ±{distance_std:.1f}m indicates inconsistent contact",
                "solution": "Practice tempo drills and work on consistent swing length. Consider lesson on swing plane.",
                "target": "Target: ±10m or less for good consistency"
            })
    
    # Direction Accuracy
    if 'Launch Direction' in club_df.columns and club_df['Launch Direction'].notna().any():
        direction_std = club_df['Launch Direction'].std()
        if direction_std > 15:
            priority_areas.append("Direction Control")
            recommendations.append({
                "icon": "🎯",
                "title": "Improve Direction Control",
                "issue": f"Direction spread of ±{direction_std:.1f}° suggests alignment or face control issues",
                "solution": "Check setup alignment and practice square club face at impact. Use alignment sticks during practice.",
                "target": "Target: ±10° for good accuracy"
            })
    
    # Launch Conditions
    if 'Launch Angle' in club_df.columns and club_df['Launch Angle'].notna().any():
        avg_launch = club_df['Launch Angle'].mean()
        if 'Driver' in title_suffix:
            if avg_launch < 10 or avg_launch > 20:
                recommendations.append({
                    "icon": "📐",
                    "title": "Optimize Launch Angle",
                    "issue": f"Driver launch angle of {avg_launch:.1f}° is outside optimal range",
                    "solution": "Adjust tee height and ball position. Consider loft adjustment or different driver.",
                    "target": "Target: 12-17° for optimal carry"
                })
        elif any(iron in title_suffix for iron in ['Iron']):
            if avg_launch < 12 or avg_launch > 30:
                recommendations.append({
                    "icon": "📐",
                    "title": "Optimize Launch Angle",
                    "issue": f"Iron launch angle of {avg_launch:.1f}° could be optimized",
                    "solution": "Check ball position and angle of attack. Ball should be slightly forward of center for irons.",
                    "target": "Target: 15-25° for optimal trajectory"
                })
    
    # Spin Analysis
    if 'Backspin' in club_df.columns and club_df['Backspin'].notna().any():
        avg_backspin = club_df['Backspin'].mean()
        if 'Driver' in title_suffix:
            if avg_backspin > 3000:
                recommendations.append({
                    "icon": "🌪️",
                    "title": "Reduce Driver Backspin",
                    "issue": f"Backspin of {avg_backspin:.0f} rpm is too high for driver",
                    "solution": "Hit up on the ball more (positive attack angle). Tee ball higher and move it forward in stance.",
                    "target": "Target: 2000-2500 rpm for maximum distance"
                })
        elif any(iron in title_suffix for iron in ['Iron']):
            if avg_backspin < 4000 or avg_backspin > 8000:
                recommendations.append({
                    "icon": "🌪️",
                    "title": "Optimize Iron Backspin",
                    "issue": f"Iron backspin of {avg_backspin:.0f} rpm is outside optimal range",
                    "solution": "Work on clean contact and proper descending blow with irons.",
                    "target": "Target: 5000-7000 rpm for good stopping power"
                })
    
    # Display recommendations
    if recommendations:
        if priority_areas:
            st.error(f"🚨 **Priority Areas**: {', '.join(priority_areas)}")
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{rec['icon']} {rec['title']}", expanded=i<2):  # Expand first 2
                st.markdown(f"**Issue:** {rec['issue']}")
                st.markdown(f"**Solution:** {rec['solution']}")
                st.markdown(f"**{rec['target']}**")
    else:
        st.success("🎉 **Excellent Performance!** Your metrics are all within optimal ranges. Keep up the great work!")
        st.info("💡 **Maintenance Tips:** Continue regular practice to maintain consistency. Focus on small refinements rather than major changes.")
